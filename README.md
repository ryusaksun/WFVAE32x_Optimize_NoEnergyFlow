# WFVAE32x_Optimize_NoEnergyFlow

基于 [WF-VAE (CVPR 2025)](https://arxiv.org/abs/2411.17459) 的 **32× 图像 VAE** 实验分支。

此分支在原作的 Haar 小波能量流架构上做了三项工程改造，并固定一条实验路线：

1. **压缩比提升到 32×**：5 个 `base_channels` 阶段（`[256, 512, 512, 1024, 1024]`），W^(1) Haar WT 贡献 2×、4 个下采样阶段贡献 16×。
2. **DCAE 空间转换 + 非参残差捷径**：`Conv + pixel_unshuffle / pixel_shuffle` 替代经典的 stride=2 / interp+conv，并叠加 `reshape-group-mean / repeat_interleave` 的零参捷径；端到端配套 HunyuanVAE 风格的 I/O 捷径（bottleneck 前/后的通道组平均 & 重复）。
3. **关闭中间层小波能量流（`use_energy_flow=false`）**：去掉 W^(2)/W^(3)… 中间层的 `in_flow_conv / out_flow_conv / 小波级联`，仅保留编码器入口的 Haar WT 与解码器出口的 IWT（即论文里的 W^(1)），压缩比不变。保留原 `use_energy_flow` 开关，便于对纸面原形式做 A/B。

配套改进：**ECA 通道注意力**（每个 ResBlock 末端 5 参数）、**多尺度判别器 + 谱归一化 + 特征匹配损失**、**LayerNorm**（修复 GroupNorm + pixel_shuffle 在 32× GAN 阶段的迁移彩色伪影）、训练稳定性修复（adaptive_weight 正确回退、EMA key 对齐）。

---

## 仓库布局

```
├── causalimagevae/
│   ├── model/
│   │   ├── vae/modeling_wfvae2.py        # Encoder/Decoder/WFDown(Classic)/WFUp(Classic)/WFIVAE2Model
│   │   ├── modules/
│   │   │   ├── wavelet.py                # Haar WT / IWT
│   │   │   ├── updownsample.py           # Classic stride-2/interp 变体使用
│   │   │   ├── resnet_block.py           # ResnetBlock2D (末端可选 ECA)
│   │   │   └── eca.py                    # ECA 通道注意力
│   │   ├── losses/
│   │   │   ├── perceptual_loss_2d.py     # LPIPSWithDiscriminator
│   │   │   ├── discriminator.py          # 单尺度 PatchGAN（legacy）
│   │   │   └── multiscale_discriminator.py   # pix2pixHD 多尺度判别器
│   │   ├── ema_model.py
│   │   └── registry.py
│   └── dataset/
│       ├── image_dataset.py / manifest_dataset.py
│       └── ddp_sampler.py
├── examples/
│   └── wfivae2-image-32x-192bc.json      # 唯一出厂配置（DCAE + no-ef + ECA + LN）
├── scripts/
│   ├── test_imagevae.py                  # forward/backward 烟雾测试
│   ├── recon_single_image.py             # 单图重建
│   ├── draw_architecture.py              # 架构图生成（DCAE + no-ef）
│   ├── save_hf_model.py                  # 转 HuggingFace 格式
│   ├── merge_encoder_and_decoder.py
│   └── resize_sa1b_256.py / resize_sa1b_512.py / resize_faces_to_256_512.py
├── architecture_diagram.html             # Mermaid 交互式架构图
├── wfvae2_32x_dcae_noef_architecture.png # 架构总图
├── wfvae2_32x_dcae_noef_blocks_detail.png# WFDownBlock / WFUpBlock 细节图
├── train_image_ddp.py                    # DDP 训练主入口
└── train_wfivae.sh                       # 训练 Shell 封装
```

---

## 环境

```bash
conda create -n wfvae python=3.10 -y
conda activate wfvae
pip install -r requirements.txt
```

依赖 PyTorch ≥ 2.x（自备 CUDA wheel）、torchvision、lpips、diffusers、accelerate、matplotlib、imageio 等；详见 `requirements.txt`。

---

## 训练

默认 shell 已把所有当前实验关键开关都选好（DCAE、关闭中间层能量流、ECA、多尺度判别器 + SN、LayerNorm）：

```bash
# 默认 2 卡 (GPU=0,1)、256px、per-GPU batch 8
bash train_wfivae.sh

# 多卡
GPU=0,1,2,3 bash train_wfivae.sh

# 切到 1024px（自动降低 batch 并开 4 步梯度累积）
RESOLUTION=1024 BATCH_SIZE=1 EPOCHS=500 EVAL_STEPS=500 bash train_wfivae.sh

# 断点续训
RESUME_CKPT=/path/to/checkpoint.ckpt bash train_wfivae.sh

# 判别器消融（默认 multiscale + sn）
DISC_TYPE=single DISC_NORM=bn bash train_wfivae.sh
DISC_TYPE=multiscale FEAT_MATCH_WEIGHT=0 bash train_wfivae.sh
```

`train_wfivae.sh` 会根据 JSON 的开关自动给 `EXP_NAME` 拼上 `_{disc_type}_{disc_norm}`、可选的 `_eca`、`_classic`、`_noef`，**把不同开关的 checkpoint 隔离在不同目录下**，避免 state_dict 互相覆盖。出厂配置下默认 EXP_NAME 为 `{project}_multiscale_sn_eca_noef`。

也可以直接用 `torchrun` 起（需要手工给数据 manifest）：

```bash
torchrun --nproc_per_node=8 train_image_ddp.py \
    --exp_name WFIVAE_1024 \
    --image_path /path/to/train_manifest.jsonl --use_manifest \
    --eval_image_path /path/to/val_manifest.jsonl \
    --model_name WFIVAE2 \
    --model_config examples/wfivae2-image-32x-192bc.json \
    --resolution 1024 --batch_size 1 --lr 1e-5 \
    --ema --wavelet_loss --eval_lpips
```

### 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model_name` | `WFIVAE2` | 注册名 |
| `--model_config` | `examples/wfivae2-image-32x-192bc.json` | 本分支唯一出厂配置 |
| `--disc_cls` | `causalimagevae.model.losses.LPIPSWithDiscriminator` | 重建 + 感知 + KL + GAN + 小波 loss |
| `--disc_start` | 80000 | 仅门控**生成器**消费 GAN/FM loss 的 step；判别器从 step 0 起就在训练 |
| `--disc_type` | `multiscale` | `single` / `multiscale`；FM loss 仅在 multiscale 下生效 |
| `--disc_norm` | `sn` | `sn` / `bn` / `in` / `none` |
| `--num_D` | 3 | multiscale 判别器尺度数 |
| `--feat_match_weight` | 10.0 | pix2pixHD 特征匹配 loss 权重 |
| `--kl_weight` | 1e-6 | |
| `--wavelet_weight` | 0.1 | 仅在 `use_energy_flow=true` 时有效（no-ef 下自动回退为 0） |
| `--disc_weight` | 0.5 | |
| `--perceptual_weight` | 1.0 | |
| `--adaptive_weight_clamp` | 1e6 | adaptive `d_weight` 上限 |

训练输出：

- `{ckpt_dir}/{exp_name}/README.md` — 运行时环境、Git 信息、配置快照（断点续训时追加“第 N 次运行”小节）
- `{exp_name}.csv` — 全部 loss/指标
- `training_curves*.png` — 7×3 子图，带 `disc_start` 标记
- `val_images/original/` & `val_images/reconstructed/` — 验证集可视化
- `checkpoint-*.ckpt` — 生成器、判别器、优化器、EMA、sampler 状态

> ⚠️ 历史原因，ckpt 里判别器的 key 是 `dics_model`（拼写错误），**不要改**，否则老 ckpt 恢复会断。

---

## 推理

### 烟雾测试（forward + backward）

```bash
python scripts/test_imagevae.py --config examples/wfivae2-image-32x-192bc.json
```

### 单图重建

```bash
python scripts/recon_single_image.py \
    --model_name WFIVAE2 \
    --from_pretrained /path/to/model \
    --image_path input.jpg --rec_path output.jpg
```

### 其他脚本

- `scripts/save_hf_model.py` — 转换 ckpt 为 HuggingFace 格式
- `scripts/merge_encoder_and_decoder.py` — 合并分离训练的 encoder / decoder
- `scripts/recon_single_image_flux.py` — FLUX.2 VAE 对比基线
- `scripts/draw_architecture.py` — 重新生成架构图 PNG

---

## 模型配置

仓库**只发货一套** JSON：`examples/wfivae2-image-32x-192bc.json`（DCAE + no-ef）。

```jsonc
{
  "latent_dim": 64,
  "base_channels": [256, 512, 512, 1024, 1024],
  "encoder_num_resblocks": [4, 4, 4, 2],
  "decoder_num_resblocks": [5, 5, 5, 3],
  "block_type": "dcae",
  "norm_type": "layernorm",
  "use_io_shortcut": true,
  "use_eca": true,
  "use_energy_flow": false
}
```

| 维度 | 值 |
|---|---|
| 压缩比 | 32×（W^(1) 2× × 4 阶段 16×） |
| Latent 形状 | 1024→[64,32,32] / 512→[64,16,16] / 256→[64,8,8] |
| 参数量 (no-ef，出厂) | Encoder 145.55 M + Decoder 322.25 M = **467.80 M** |
| 参数量 (ef=true，纸面形) | Encoder 152.17 M + Decoder 337.71 M = 489.88 M |

LayerNorm 是**不可替换**的：换回 GroupNorm 会在 32× GAN 阶段复现“迁移彩色高频伪影”，经排查是 GroupNorm 与 `pixel_shuffle` 的组合问题。

### 开关语义

| 开关 | 作用 | 与其他开关的约束 |
|---|---|---|
| `block_type` | `"dcae"`（shipped）`/` `"classic"`（A/B）—— 空间转换实现。两者的 state_dict key **不兼容**。 | Classic 模式下强制 `use_io_shortcut=False`（软降级并改写 config.json） |
| `use_io_shortcut` | HunyuanVAE 风格的 bottleneck I/O 捷径（零参） | 仅 DCAE 有意义；需满足 `base_channels[-1] % (2*latent_dim) == 0` |
| `use_eca` | 每个 ResBlock 末端的 ECA 通道注意力（参数 k=5） | 与 `block_type` 正交 |
| `use_energy_flow` | 中间层小波能量流（W^(2)/W^(3)…）；W^(1) 始终保留，压缩比不变 | 与其他开关正交，4 个开关共 2⁴ 组合，仓库只发货其一 |

**Checkpoint 不跨配置兼容**：翻 `block_type` / `use_eca` / `use_energy_flow` / `disc_type` / `disc_norm` 任意一个都会改生成器或判别器的 state_dict，`_noef` / `_eca` / `_classic` / `_{disc_type}_{disc_norm}` 后缀就是避免用错 ckpt 续训的第一道保险。

---

## 架构一眼看（32× / 1024px / DCAE / no-ef）

```
Encoder  [3, 1024]
  └ Haar WT                              → [12, 512]   （W^(1)，始终保留）
  └ conv_in (12→256)                     → [256, 512]
  └ WFDownBlock ×4  (DCAE + 捷径)         → [1024, 32] （16×）
  └ Mid: ResBlock → Attention2DFix → ResBlock
  └ (可选) I/O 捷径: reshape → group-mean
  └ norm → SiLU → conv_out (1024→128)    → [128, 32]   (+= io_shortcut)
  └ 高斯采样                              → [64, 32, 32]

Decoder  [64, 32]
  └ conv_in (64→1024) (+ 可选 I/O 捷径)   → [1024, 32]
  └ Mid: ResBlock → Attention2DFix → ResBlock
  └ WFUpBlock ×4   (DCAE + 捷径)          → [256, 512]
  └ norm → SiLU → conv_out (256→12)      → [12, 512]
  └ (若 ef=true) 最后一级 WFUpBlock 的小波残差加到 h[:, :3]
  └ Inverse Haar WT                      → [3, 1024]   （W^(1) 对称）
```

更完整的细节图见 `wfvae2_32x_dcae_noef_architecture.png` / `wfvae2_32x_dcae_noef_blocks_detail.png`，或 `architecture_diagram.html`（浏览器打开）。

---

## 数据格式

递归扫描目录（jpg/png/webp/bmp），首次启动会缓存文件列表到 pickle。也支持 JSONL manifest（`--use_manifest`）：

```json
{"image_path": "/abs/path/to/image.jpg"}
```

字段别名：`image_path` / `path` / `target`。

---

## 引用

本分支的架构主干来自：

```bibtex
@misc{li2024wfvaeenhancingvideovae,
      title={WF-VAE: Enhancing Video VAE by Wavelet-Driven Energy Flow for Latent Video Diffusion Model},
      author={Zongjian Li and Bin Lin and Yang Ye and Liuhan Chen and Xinhua Cheng and Shenghai Yuan and Li Yuan},
      year={2024},
      eprint={2411.17459},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.17459}
}
```

---

## License

Apache 2.0，见 [LICENSE](LICENSE)。

# Image VAE 训练指南（当前仓库版本）

本文档对应当前仓库的真实训练流程，推荐使用 `train_wfivae.sh` 启动。

## 1. 训练入口与核心文件

- 推荐入口：`train_wfivae.sh`
- 底层训练器：`train_image_ddp.py`
- 模型配置：`examples/wfivae2-image-1024.json`
- 图像模型代码：`causalimagevae/`

注意：仓库当前没有 `train_image_ddp.sh`，也没有 `examples/wfivae2-image.json`。

## 2. 环境准备

```bash
conda create -n wfvae python=3.10 -y
conda activate wfvae
pip install -r requirements.txt
```

## 3. 数据准备

`train_wfivae.sh` 期望你提供一个“总清单”JSONL（通过 `ORIGINAL_MANIFEST`）。脚本会自动随机划分训练/验证并生成临时清单。

清单格式（每行一个 JSON）：

```jsonl
{"image_path": "/abs/or/rel/path/to/image1.jpg"}
{"image_path": "relative/path/to/image2.png"}
```

说明：`image_path` 支持绝对或相对路径。相对路径时，`manifest` 文件所在目录会作为基准目录。

## 4. 快速开始（推荐）

```bash
cd /Users/ryuichi/Desktop/renxing/WFVAE-series/WF-VAE-8x_disc_start_5

GPU=0,1,2,3 \
ORIGINAL_MANIFEST=/path/to/all_images.jsonl \
OUTPUT_DIR=/path/to/output \
DISABLE_WANDB=1 \
RESOLUTION=1024 \
EVAL_SUBSET_SIZE=0 \
bash train_wfivae.sh
```

关键点：

- `EVAL_SUBSET_SIZE=0` 表示验证集全量（推荐用于完整 patch 分数导出）。
- 脚本会用 `torchrun` 启动 DDP。
- 训练结束或中断时会清理临时 `train_manifest.jsonl` / `eval_manifest.jsonl`。

## 5. 常用环境变量（train_wfivae.sh）

| 环境变量 | 作用 | 默认值 |
|---|---|---|
| `GPU` | 使用的 GPU 列表 | `0` |
| `ORIGINAL_MANIFEST` | 总数据清单路径 | `/mnt/goosefs-lite-mnt/image_manifest.jsonl` |
| `OUTPUT_DIR` | 输出目录 | `/mnt/sdc/WF-VAE-8x_disc_start_5_PatchGANScore_30w` |
| `DISABLE_WANDB` | 是否关闭 wandb（`1/true/yes` 关闭） | `1` |
| `WANDB_PROJECT` | wandb 项目名（仅在启用 wandb 时生效） | `WFIVAE` |
| `RESOLUTION` | 训练分辨率 | `1024` |
| `MODEL_CONFIG` | 模型配置文件 | 分辨率对应默认值 |
| `EXP_NAME` | 实验名 | 分辨率对应默认值 |
| `BATCH_SIZE` | 每卡训练 batch size | 1024: `2`, 512: `8` |
| `EVAL_BATCH_SIZE` | 验证 batch size | 1024: `2`, 512: `4` |
| `TRAIN_RATIO` | 训练集划分比例 | `0.9` |
| `MAX_STEPS` | 最大训练步数 | `100000` |
| `SAVE_CKPT_STEP` | checkpoint 间隔 | `2000` |
| `EVAL_STEPS` | 验证间隔 | `1000` |
| `EVAL_SUBSET_SIZE` | 验证样本数（`<=0` 全量） | `0` |
| `EVAL_NUM_IMAGE_LOG` | 每次验证保存图像数量（并对齐 patch 可视化样本） | `20` |
| `CSV_LOG_STEPS` | CSV 记录间隔 | `50` |
| `LOG_STEPS` | W&B/日志记录间隔 | `10` |
| `DATASET_NUM_WORKER` | dataloader worker 数 | `8` |
| `RESUME_CKPT` | 恢复训练 checkpoint | 空 |

如果要启用 wandb，请设置 `DISABLE_WANDB=0`，并先执行 `wandb login`。

## 6. 输出产物

在 `${OUTPUT_DIR}/${EXP_NAME}-lr...-bs...-rs.../` 下会生成：

- `training_losses.csv`
- `training_curves.png`（每次验证后更新）
- `training_curves_final.png`（训练结束）
- `checkpoint-*.ckpt`
- `val_images/original/` 与 `val_images/reconstructed/`
- `val_patch_scores/step_xxxxxxxx/`
- `val_patch_scores/step_xxxxxxxx_ema/`

每个 `val_patch_scores` 目录包含：

- `summary.csv`
- `patch_vis/real/*.png`（原图 + patch 热力图 + 叠加图）
- `patch_vis/recon/*.png`（原图 + 重建打分热力图 + 叠加图）

## 7. 训练监控

W&B 常见指标：

- `train/generator_loss`
- `train/discriminator_loss`
- `train/rec_loss`
- `val/psnr`
- `val/lpips`
- `val/patch_*`（patch 分数均值和直方图，含 EMA 与非 EMA）

训练曲线图当前面板为：

- `generator_loss`
- `discriminator_loss`
- `rec_loss`
- `perceptual_loss`
- `kl_loss`
- `wavelet_loss`
- `psnr`
- `lpips`
- `psnr_ema`
- `lpips_ema`

说明：

- CSV 会把验证指标拆分为 `psnr/lpips`（非 EMA）与 `psnr_ema/lpips_ema`（EMA），避免混写。

## 8. 手动 torchrun 启动（调试用）

```bash
torchrun --nproc_per_node=8 train_image_ddp.py \
  --exp_name WFIVAE2-1024-disc5 \
  --image_path /path/to/train_manifest.jsonl \
  --eval_image_path /path/to/eval_manifest.jsonl \
  --use_manifest \
  --model_name WFIVAE2 \
  --model_config examples/wfivae2-image-1024.json \
  --ckpt_dir /path/to/output \
  --resolution 1024 \
  --batch_size 2 \
  --max_steps 100000 \
  --eval_steps 1000 \
  --eval_subset_size 0 \
  --eval_lpips \
  --ema \
  --wavelet_loss
```

## 9. 常见问题

### 9.1 OOM

- 减小 `BATCH_SIZE`
- 减小 `EVAL_BATCH_SIZE`
- 使用 `--mix_precision fp16`（如硬件不适合 bf16）
- 降低 `DATASET_NUM_WORKER`

### 9.2 patch 分数没有全量输出

确认：`EVAL_SUBSET_SIZE=0` 或显式设置为 `<=0`。

### 9.3 `MAX_STEPS` 不生效

当前版本已生效；到达阈值后会打印 `Reached max_steps=...` 并停止训练。

## 10. 推理/重建

单图重建脚本：`scripts/recon_single_image.py`

```bash
python scripts/recon_single_image.py \
  --model_name WFIVAE2 \
  --from_pretrained /path/to/model \
  --image_path assets/gt_5544.jpg \
  --rec_path rec.jpg \
  --device cuda \
  --short_size 1024
```

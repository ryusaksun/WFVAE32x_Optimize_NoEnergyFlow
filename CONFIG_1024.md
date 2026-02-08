# 1024分辨率训练配置说明

## 数据路径配置

您的数据集配置：
```bash
训练集: /mnt/sda/datasets/imagevae_1024/train
验证集: /mnt/sda/datasets/imagevae_1024/eval
输出目录: /mnt/sdc/yyy_WFVAE_原版
```

## 关键参数调整（针对1024分辨率）

### 1. 批次大小
由于1024分辨率图像较大，建议：
- **每GPU批次**: 2（24GB显存）或 1（16GB显存）
- **8卡总批次**: 16 或 8

### 2. 模型配置
可能需要调整模型配置以适应1024分辨率：

```json
{
  "_class_name": "WFIVAE2Model",
  "latent_dim": 8,  // 增加潜在维度
  "base_channels": [128, 256, 512, 768],  // 增加一层
  "encoder_num_resblocks": 2,
  "encoder_energy_flow_size": 128,
  "decoder_num_resblocks": 3,
  "decoder_energy_flow_size": 128,
  "dropout": 0.0,
  "norm_type": "layernorm",
  "mid_layers_type": ["ResnetBlock2D", "Attention2DFix", "ResnetBlock2D"]
}
```

### 3. 内存优化建议

如果遇到OOM（内存不足）：

```bash
# 选项1: 减小批次
--batch_size 1

# 选项2: 使用FP16混合精度
--mix_precision fp16

# 选项3: 减少验证批次
--eval_batch_size 2

# 选项4: 禁用LPIPS评估（训练时）
# 移除 --eval_lpips 参数

# 选项5: 减少数据加载worker
--dataset_num_worker 2
```

## 启动方式

### 方式1: 使用配置好的脚本（推荐）
```bash
bash train_image_ddp.sh
```

### 方式2: 使用交互式快速启动
```bash
bash quick_start_1024.sh
# 会提示选择GPU数量
```

### 方式3: 手动启动（调试用）
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=WFIVAE_1024

torchrun --nproc_per_node=8 train_image_ddp.py \
    --exp_name WFIVAE_1024 \
    --image_path /mnt/sda/datasets/imagevae_1024/train \
    --eval_image_path /mnt/sda/datasets/imagevae_1024/eval \
    --ckpt_dir /mnt/sdc/yyy_WFVAE_原版 \
    --resolution 1024 \
    --batch_size 2 \
    --model_name WFIVAE2 \
    --model_config examples/wfivae2-image.json \
    [其他参数...]
```

## Manifest文件说明

您提到了manifest文件：
- `train_manifest.jsonl`
- `eval_manifest.jsonl`

当前的`ImageDataset`会自动扫描目录中的图像文件。如果您需要使用manifest文件进行更精细的控制（如元数据、标签、条件生成），我可以为您扩展数据集类。

Manifest文件格式示例：
```jsonl
{"image_path": "train/img001.jpg", "caption": "描述", "metadata": {...}}
{"image_path": "train/img002.png", "caption": "描述", "metadata": {...}}
```

## 性能预估（1024分辨率）

| 配置 | 速度 | 显存/卡 | 说明 |
|------|------|---------|------|
| 8×A100(80GB), bs=2 | ~0.5 it/s | ~40GB | 推荐配置 |
| 8×A100(40GB), bs=1 | ~0.8 it/s | ~22GB | 可用 |
| 8×V100(32GB), bs=1, fp16 | ~0.6 it/s | ~18GB | 需要FP16 |

## 训练时间估算

假设：
- 数据集: 100万张图像
- 配置: 8卡，batch_size=2/卡，总batch=16
- 速度: 0.5 it/s

计算：
- 每个epoch: 1,000,000 / 16 = 62,500 iterations
- 时间/epoch: 62,500 / 0.5 = 125,000秒 ≈ 35小时
- 10 epochs: ~350小时 ≈ 14.5天

## 监控指标

训练过程中关注：
1. **重建损失** (rec_loss): 应该持续下降，目标 < 0.1
2. **PSNR**: 验证集PSNR，目标 > 25dB
3. **LPIPS**: 感知损失，目标 < 0.15
4. **判别器损失**: 应该在生成器损失附近平衡

## 检查数据集

训练前验证数据：
```bash
# 检查图像数量
find /mnt/sda/datasets/imagevae_1024/train -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l

# 检查图像尺寸（示例）
identify /mnt/sda/datasets/imagevae_1024/train/img001.jpg | grep 1024x1024

# 测试数据加载
python3 -c "
from causalimagevae.dataset.image_dataset import ImageDataset
ds = ImageDataset('/mnt/sda/datasets/imagevae_1024/train', resolution=1024)
print(f'Dataset size: {len(ds)}')
sample = ds[0]
print(f'Image shape: {sample[\"image\"].shape}')
"
```

## 常见问题

### 1. 训练很慢
- 检查数据加载：增加 `--dataset_num_worker`
- 使用更快的存储（SSD而非HDD）
- 确认混合精度启用：`--mix_precision bf16`

### 2. 重建质量不好
- 增加训练步数
- 调整 `--perceptual_weight` (试试 2.0)
- 确保 `--wavelet_loss` 启用
- 降低 `--disc_start` (提前开始判别器训练)

### 3. 训练不稳定
- 增大 `--disc_start` 到 10000
- 降低学习率 `--lr 5e-6`
- 使用梯度裁剪 `--clip_grad_norm 1.0`

## 下一步

训练完成后，checkpoint保存在：
```
/mnt/sdc/yyy_WFVAE_原版/WFIVAE_1024-lr1.00e-05-bs2-rs1024/
    ├── checkpoint-5000.ckpt
    ├── checkpoint-10000.ckpt
    └── ...
```

使用模型进行推理请参考 `README_image_training.md` 中的推理部分。

# 快速参考卡 - 1024分辨率训练

## 📁 您的配置

```bash
训练数据: /mnt/sda/datasets/imagevae_1024/train
验证数据: /mnt/sda/datasets/imagevae_1024/eval
输出目录: /mnt/sdc/yyy_WFVAE_原版
分辨率:   1024x1024
```

## 🚀 三种启动方式

### 1. 标准启动（推荐）
```bash
cd /media/HDU/yyy/WF-VAE-main原版
bash train_image_ddp.sh
```

### 2. 交互式启动
```bash
bash quick_start_1024.sh
# 会提示选择GPU数量
```

### 3. 手动调试
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 train_image_ddp.py \
    --exp_name test_run \
    --image_path /mnt/sda/datasets/imagevae_1024/train \
    --resolution 1024 \
    --batch_size 1 \
    [...]
```

## 📊 重要文件

| 文件 | 用途 |
|------|------|
| [train_image_ddp.sh](train_image_ddp.sh) | 主训练脚本 ✅ 已配置 |
| [quick_start_1024.sh](quick_start_1024.sh) | 交互式启动 |
| [CONFIG_1024.md](CONFIG_1024.md) | 详细配置说明 |
| [USE_MANIFEST.md](USE_MANIFEST.md) | Manifest文件使用（可选） |
| [examples/wfivae2-image-1024.json](examples/wfivae2-image-1024.json) | 1024模型配置 |

## ⚙️ 关键参数（已优化1024分辨率）

```bash
--resolution 1024           # 图像分辨率
--batch_size 2              # 每GPU批次（24GB显存）
--lr 1e-5                   # 学习率
--disc_start 5000           # 判别器开始步数
--mix_precision bf16        # 混合精度训练
--eval_lpips                # 评估LPIPS指标
--ema                       # 使用EMA
--wavelet_loss              # 使用小波损失
```

## 🔍 检查数据集

```bash
# 查看训练集大小
find /mnt/sda/datasets/imagevae_1024/train -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l

# 查看验证集大小  
find /mnt/sda/datasets/imagevae_1024/eval -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l

# 测试数据加载
python3 << 'PYTHON'
from causalimagevae.dataset.image_dataset import ImageDataset
ds = ImageDataset('/mnt/sda/datasets/imagevae_1024/train', resolution=1024)
print(f'✓ Dataset size: {len(ds)} images')
sample = ds[0]
print(f'✓ Image shape: {sample["image"].shape}')
PYTHON
```

## 📈 监控训练

### WandB 配置
```bash
export WANDB_PROJECT=WFIVAE_1024
export WANDB_API_KEY=your_key_here  # 可选
```

### 关键指标
- **rec_loss**: 重建损失（目标 < 0.1）
- **val/psnr**: PSNR（目标 > 25dB）
- **val/lpips**: 感知损失（目标 < 0.15）
- **train/latents_std**: 潜在编码标准差

## 💾 Checkpoint 位置

```
/mnt/sdc/yyy_WFVAE_原版/
└── WFIVAE_1024-lr1.00e-05-bs2-rs1024/
    ├── checkpoint-5000.ckpt
    ├── checkpoint-10000.ckpt
    └── ...
```

## 🛠️ 常见调整

### 内存不足 (OOM)
```bash
--batch_size 1              # 减小批次
--mix_precision fp16        # 使用FP16
--eval_batch_size 2         # 减小验证批次
--dataset_num_worker 2      # 减少worker
```

### 训练太慢
```bash
--dataset_num_worker 8      # 增加worker
--eval_steps 2000           # 减少评估频率
--eval_subset_size 200      # 减少验证样本
```

### 重建质量差
```bash
--perceptual_weight 2.0     # 增加感知损失
--disc_start 3000           # 提前开始判别器
--wavelet_weight 0.2        # 增加小波损失
```

## 🔧 使用 Manifest 文件（可选）

如果您想使用提供的 manifest 文件：

```bash
# 1. 检查 manifest 文件
head /mnt/sda/datasets/imagevae_1024/train_manifest.jsonl

# 2. 查看 USE_MANIFEST.md 获取详细说明
cat USE_MANIFEST.md
```

## 📞 遇到问题？

1. 检查 CUDA 可用性:
```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

2. 检查模块导入:
```bash
cd /media/HDU/yyy/WF-VAE-main原版
python3 -c "from causalimagevae.model.vae import WFIVAE2Model; print('✓ Import OK')"
```

3. 查看日志:
```bash
# 训练日志会输出到终端
# WandB 日志: https://wandb.ai/your-project/WFIVAE_1024
```

## 📚 更多文档

- [README_image_training.md](README_image_training.md) - 完整训练指南
- [CONFIG_1024.md](CONFIG_1024.md) - 1024配置详解
- [CONVERSION_SUMMARY.md](CONVERSION_SUMMARY.md) - 转换总结

---

**准备好了？运行这个命令开始训练:**

```bash
bash train_image_ddp.sh
```

🎯 预计训练时间: 约14-20天 (8×A100, 100万图像, 10 epochs)

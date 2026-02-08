# Image VAE Training Guide

本指南说明如何使用 `causalimagevae` 训练图像 VAE 模型。

## 环境准备

确保您已安装以下依赖：
```bash
pip install torch torchvision diffusers einops lpips wandb pillow
```

## 文件说明

- `train_image_ddp.py` - 分布式训练脚本（图像版本）
- `train_image_ddp.sh` - 训练启动脚本
- `causalimagevae/` - 图像 VAE 模块
- `examples/wfivae2-image.json` - 模型配置文件

## 快速开始

### 1. 准备数据集

将您的图像放在一个目录中，支持的格式：`jpg`, `jpeg`, `png`, `webp`, `bmp`

```
/path/to/your/images/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

### 2. 修改训练脚本

编辑 `train_image_ddp.sh`，更新以下参数：

```bash
--image_path /path/to/your/images \
--eval_image_path /path/to/your/validation/images \
--model_config examples/wfivae2-image.json \
```

### 3. 启动训练

```bash
# 单机 8 卡训练
bash train_image_ddp.sh

# 或者使用 torchrun 直接运行
torchrun --nproc_per_node=8 train_image_ddp.py \
    --exp_name WFIVAE_test \
    --image_path /path/to/images \
    --eval_image_path /path/to/val_images \
    --model_name WFIVAE2 \
    --model_config examples/wfivae2-image.json \
    --resolution 256 \
    --batch_size 4 \
    --lr 1e-5 \
    --epochs 10 \
    --eval_lpips \
    --ema \
    --wavelet_loss
```

## 主要参数说明

### 数据参数
- `--image_path` - 训练图像路径
- `--eval_image_path` - 验证图像路径
- `--resolution` - 图像分辨率（默认256）
- `--batch_size` - 批次大小

### 模型参数
- `--model_name` - 模型名称（使用 `WFIVAE2`）
- `--model_config` - 模型配置文件路径
- `--latent_dim` - 潜在空间维度（在配置文件中设置）

### 训练参数
- `--lr` - 学习率（默认 1e-5）
- `--epochs` - 训练轮数
- `--mix_precision` - 混合精度（bf16/fp16/fp32）
- `--ema` - 使用 EMA
- `--ema_decay` - EMA 衰减率（默认 0.999）

### 损失参数
- `--disc_start` - 判别器开始训练的步数（默认 5000）
- `--disc_weight` - 判别器损失权重（默认 0.5）
- `--kl_weight` - KL 散度权重（默认 1e-6）
- `--perceptual_weight` - 感知损失权重（默认 1.0）
- `--wavelet_loss` - 使用小波损失
- `--wavelet_weight` - 小波损失权重（默认 0.1）

### 验证参数
- `--eval_steps` - 验证间隔步数（默认 1000）
- `--eval_batch_size` - 验证批次大小
- `--eval_subset_size` - 验证样本数量
- `--eval_lpips` - 计算 LPIPS 指标

## 模型配置

`examples/wfivae2-image.json` 配置示例：

```json
{
  "_class_name": "WFIVAE2Model",
  "latent_dim": 4,
  "base_channels": [128, 256, 512],
  "encoder_num_resblocks": 2,
  "encoder_energy_flow_size": 128,
  "decoder_num_resblocks": 3,
  "decoder_energy_flow_size": 128,
  "dropout": 0.0,
  "norm_type": "layernorm",
  "mid_layers_type": ["ResnetBlock2D", "Attention2DFix", "ResnetBlock2D"]
}
```

## 架构说明

### CausalImageVAE 与 CausalVideoVAE 的区别

| 特性 | CausalVideoVAE | CausalImageVAE |
|------|---------------|----------------|
| 输入维度 | (B, 3, T, H, W) | (B, 3, H, W) |
| Haar 变换 | 3D (24 通道) | 2D (12 通道) |
| 卷积类型 | CausalConv3d | Conv2d |
| 采样 | 时空采样 | 空间采样 |
| 注意力 | Attention3D | Attention2D |

### 模型组件
1. **WFDownBlock** - 带能量流的下采样块（2D Haar + 空间下采样）
2. **WFUpBlock** - 带能量流的上采样块（2D Haar + 空间上采样）
3. **Encoder** - 编码器（Conv2d + ResnetBlock2D + Attention2D）
4. **Decoder** - 解码器（对称结构）

## 监控和日志

训练过程会记录到 WandB：

```python
export WANDB_PROJECT=WFIVAE
```

主要监控指标：
- `train/generator_loss` - 生成器损失
- `train/discriminator_loss` - 判别器损失
- `train/rec_loss` - 重建损失
- `val/psnr` - 峰值信噪比
- `val/lpips` - 感知损失

## Checkpoint 管理

Checkpoint 保存在 `--ckpt_dir` 指定的目录：

```
./results/WFIVAE-lr1.00e-05-bs4-rs256/
    ├── checkpoint-1000.ckpt
    ├── checkpoint-2000.ckpt
    └── ...
```

### 从 Checkpoint 恢复训练

```bash
torchrun --nproc_per_node=8 train_image_ddp.py \
    --resume_from_checkpoint ./results/xxx/checkpoint-1000.ckpt \
    [其他参数...]
```

## 推理使用

训练完成后，可以使用模型进行推理：

```python
from causalimagevae.model.vae import WFIVAE2Model
import torch
from PIL import Image
import torchvision.transforms as T

# 加载模型
model = WFIVAE2Model.from_pretrained("path/to/checkpoint")
model.eval()
model.cuda()

# 准备图像
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

image = Image.open("test.jpg")
img_tensor = transform(image).unsqueeze(0).cuda()

# 编码
with torch.no_grad():
    latent = model.encode(img_tensor).latent_dist.sample()
    print(f"Latent shape: {latent.shape}")
    
    # 解码
    reconstructed = model.decode(latent).sample
    
    # 保存重建图像
    reconstructed = (reconstructed + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    T.ToPILImage()(reconstructed[0]).save("reconstructed.jpg")
```

## 常见问题

### 1. 内存不足
- 减小 `--batch_size`
- 减小 `--resolution`
- 使用 `--mix_precision fp16`

### 2. 训练不稳定
- 增大 `--disc_start`（延迟判别器训练）
- 减小 `--disc_weight`
- 调整 `--lr`

### 3. 重建质量差
- 增加 `--perceptual_weight`
- 启用 `--wavelet_loss`
- 增加训练步数

## 参考资料

- 原始论文：WF-VAE: Wavelet Flow VAE
- GitHub: [WF-VAE-main](https://github.com/xxx/WF-VAE)

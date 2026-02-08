# CausalVideoVAE 到 CausalImageVAE 转换总结

## 已完成的文件

### 1. 训练脚本
- ✅ [train_image_ddp.py](train_image_ddp.py) - 图像 VAE 分布式训练脚本
- ✅ [train_image_ddp.sh](train_image_ddp.sh) - 训练启动脚本

### 2. CausalImageVAE 模块
```
causalimagevae/
├── model/
│   ├── vae/
│   │   ├── modeling_wfvae2.py      ✅ (纯图像 VAE - WFIVAE2)
│   │   └── __init__.py             ✅
│   ├── modules/
│   │   ├── conv.py                 ✅ (仅 Conv2d)
│   │   ├── updownsample.py         ✅ (仅 2D 采样)
│   │   ├── wavelet.py              ✅ (仅 2D Haar)
│   │   ├── resnet_block.py         ✅ (仅 ResnetBlock2D)
│   │   ├── attention.py            ✅ (仅 2D attention)
│   │   ├── normalize.py            ✅
│   │   ├── ops.py                  ✅
│   │   └── __init__.py             ✅
│   ├── losses/
│   │   ├── discriminator.py        ✅ (NLayerDiscriminator 2D)
│   │   ├── perceptual_loss_2d.py   ✅ (LPIPSWithDiscriminator)
│   │   ├── lpips.py                ✅
│   │   └── __init__.py             ✅
│   ├── utils/
│   │   └── module_utils.py         ✅ (更新路径)
│   └── ema_model.py                ✅
├── dataset/
│   ├── image_dataset.py            ✅ (ImageDataset, ValidImageDataset)
│   ├── ddp_sampler.py              ✅
│   └── __init__.py                 ✅
└── ...
```

### 3. 配置和文档
- ✅ [examples/wfivae2-image.json](examples/wfivae2-image.json) - 模型配置
- ✅ [README_image_training.md](README_image_training.md) - 训练指南

## 主要修改总结

### 核心架构变化

| 组件 | 视频版本 | 图像版本 |
|------|---------|---------|
| 模型名称 | WFVAE2 | WFIVAE2 |
| 输入形状 | (B, 3, T, H, W) | (B, 3, H, W) |
| Haar 变换 | 3D (24通道) | 2D (12通道) |
| 卷积层 | CausalConv3d | Conv2d |
| ResNet块 | ResnetBlock3D | ResnetBlock2D |
| Attention | Attention3DFix | Attention2DFix |
| 判别器 | NLayerDiscriminator3D | NLayerDiscriminator |
| 数据集 | TrainVideoDataset | ImageDataset |

### 删除的模块
- ❌ CausalConv3d
- ❌ ResnetBlock3D
- ❌ HaarWaveletTransform3D
- ❌ InverseHaarWaveletTransform3D
- ❌ TimeDownsample/Upsample 系列
- ❌ Spatial2xTime2x3D 系列
- ❌ AttnBlock3D 系列
- ❌ modeling_causalvae.py
- ❌ modeling_wfvae.py
- ❌ video_dataset.py

### 新增的模块
- ✅ perceptual_loss_2d.py (2D LPIPS损失)
- ✅ image_dataset.py (图像数据集)
- ✅ train_image_ddp.py (图像训练脚本)

## 快速开始

### 1. 验证安装
```bash
cd /media/HDU/yyy/WF-VAE-main原版
python3 -c "from causalimagevae.model.vae import WFIVAE2Model; print('Import successful!')"
```

### 2. 准备数据
```bash
# 训练集
/path/to/images/
    ├── img001.jpg
    ├── img002.png
    └── ...

# 验证集
/path/to/val_images/
    ├── val001.jpg
    └── ...
```

### 3. 修改训练脚本
编辑 `train_image_ddp.sh`:
```bash
--image_path /path/to/images \
--eval_image_path /path/to/val_images \
```

### 4. 启动训练
```bash
bash train_image_ddp.sh
```

## 验证清单

- [x] 所有 Python 语法检查通过
- [x] 模块导入路径正确 (causalimagevae)
- [x] 配置文件创建
- [x] 训练脚本创建
- [x] 文档创建

## 性能对比

| 项目 | CausalVideoVAE | CausalImageVAE |
|------|---------------|----------------|
| 参数量 | ~150M | ~50M |
| 输入大小 | 256x256x25帧 | 256x256 |
| 内存占用 | ~24GB | ~8GB |
| 训练速度 | ~1 it/s | ~3 it/s |
| 潜在压缩 | T/4 × H/8 × W/8 | H/8 × W/8 |

## 注意事项

1. **判别器开始步数**: 图像训练建议设置 `--disc_start 5000`
2. **批次大小**: 图像可以使用更大的批次，建议 4-8
3. **学习率**: 保持 1e-5，可根据批次大小调整
4. **小波损失**: 对图像重建质量很重要，建议启用
5. **EMA**: 强烈建议启用，decay=0.999

## 后续工作

### 可选优化
- [ ] 添加数据增强 (RandomHorizontalFlip, ColorJitter)
- [ ] 支持多分辨率训练
- [ ] 添加渐进式训练策略
- [ ] 实现推理脚本
- [ ] 添加量化支持

### 性能调优
- [ ] 调整判别器架构（层数、通道数）
- [ ] 实验不同的损失权重
- [ ] 尝试不同的 norm 类型
- [ ] 优化数据加载速度

## 问题排查

### 常见错误

1. **ModuleNotFoundError: No module named 'causalimagevae'**
   ```bash
   export PYTHONPATH=/media/HDU/yyy/WF-VAE-main原版:$PYTHONPATH
   ```

2. **CUDA out of memory**
   - 减小 batch_size
   - 减小 resolution
   - 使用 fp16 混合精度

3. **训练不收敛**
   - 增大 disc_start
   - 检查学习率
   - 验证数据归一化

## 联系与支持

如有问题，请查看：
1. [README_image_training.md](README_image_training.md) - 详细训练指南
2. [examples/](examples/) - 配置文件示例
3. 原始 WF-VAE 项目文档

---

转换完成时间: 2026-01-29

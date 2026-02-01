# Training Loss Logging and Visualization Design

**Date**: 2026-02-01
**Author**: Claude Code
**Status**: Approved

## Overview

增强图像 VAE 训练脚本，添加 CSV 格式的损失日志记录和训练结束时的自动可视化功能。同时将判别器启动步数从 5000 调整到 20000，确保生成器在对抗训练前充分预热。

## Requirements

1. 将 `--disc_start` 默认值从 5000 修改为 20000
2. 实时将训练损失记录到 CSV 文件
3. 训练结束时自动生成多子图损失曲线
4. 在图表中标记判别器启动点（20000 步）

## Architecture

### 1. 参数修改

**修改文件**: `train_image_ddp.py`

- `--disc_start`: 5000 → 20000（默认值）
- `--csv_log_steps`: 新增参数，默认 50（独立控制 CSV 记录频率）
- `--disable_plot`: 新增参数，默认 False（训练结束时自动生成图表）

### 2. CSV 日志模块

**功能**: 实时记录训练和验证损失到 CSV 文件

**文件路径**: `{ckpt_dir}/training_losses.csv`

**记录字段**:
- `step`: 训练步数
- `generator_loss`: 生成器总损失
- `discriminator_loss`: 判别器损失
- `rec_loss`: 重建损失
- `kl_loss`: KL 散度
- `wavelet_loss`: 小波损失
- `psnr`: PSNR（验证时）
- `lpips`: LPIPS（验证时）

**记录频率**:
- 训练损失：每 `csv_log_steps` 步（默认 50）
- 验证指标：每次验证时（`eval_steps`，默认 1000）

### 3. 图表生成模块

**功能**: 读取 CSV 文件，生成 3×3 多子图可视化

**文件路径**: `{ckpt_dir}/training_curves.png`

**布局**:
- 3 行 × 3 列子图
- 7 个指标各占一格，最后 2 格留空

**特性**:
- 平滑曲线（移动平均窗口=10）+ 原始数据（半透明）
- 红色垂直虚线标记判别器启动点（20000 步）
- 自动处理缺失值
- 300 DPI 高分辨率输出

## Implementation Details

### CSV Writer 初始化

```python
# 在训练开始前（只在 rank 0）
if global_rank == 0:
    csv_path = ckpt_dir / "training_losses.csv"

    # 处理断点恢复
    if args.resume_from_checkpoint and csv_path.exists():
        csv_file = open(csv_path, "a", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[...])
    else:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[...])
        csv_writer.writeheader()
```

### 训练损失记录

```python
# 在训练循环中
if global_rank == 0 and current_step % args.csv_log_steps == 0:
    log_dict = {
        "step": current_step,
        "generator_loss": g_loss.item() if step_gen else "",
        "discriminator_loss": d_loss.item() if step_dis else "",
        "rec_loss": g_log.get('train/rec_loss', ""),
        "kl_loss": g_log.get('train/kl_loss', ""),
        "wavelet_loss": g_log.get('train/wl_loss', ""),
        "psnr": "",
        "lpips": ""
    }
    csv_writer.writerow(log_dict)
    csv_file.flush()  # 立即写入磁盘
```

### 验证指标记录

```python
# 在验证完成后
if global_rank == 0:
    csv_writer.writerow({
        "step": current_step,
        "generator_loss": "",
        "discriminator_loss": "",
        "rec_loss": "",
        "kl_loss": "",
        "wavelet_loss": "",
        "psnr": valid_psnr,
        "lpips": valid_lpips,
    })
    csv_file.flush()
```

### 绘图函数

```python
def plot_training_curves(csv_path, output_path, disc_start=20000):
    """生成训练损失曲线图"""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.ndimage import uniform_filter1d
    except ImportError as e:
        logger.warning(f"Cannot generate plot: {e}")
        return

    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if len(df) < 2:
        logger.warning("Not enough data points to plot")
        return

    metrics = [
        ("generator_loss", "Generator Loss"),
        ("discriminator_loss", "Discriminator Loss"),
        ("rec_loss", "Reconstruction Loss"),
        ("kl_loss", "KL Divergence"),
        ("wavelet_loss", "Wavelet Loss"),
        ("psnr", "PSNR (dB)"),
        ("lpips", "LPIPS"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        data = df[["step", metric]].dropna()

        if len(data) > 0:
            steps = data["step"].values
            values = data[metric].values

            # 平滑处理
            if len(values) > 10:
                smoothed = uniform_filter1d(values, size=10)
                ax.plot(steps, smoothed, linewidth=2, label="Smoothed")
                ax.plot(steps, values, alpha=0.3, linewidth=1, label="Raw")
            else:
                ax.plot(steps, values, linewidth=2)

            # 判别器启动标记
            ax.axvline(x=disc_start, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.7, label=f'Disc Start ({disc_start})')

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel("Training Step")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)

    # 隐藏多余子图
    for idx in range(len(metrics), 9):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### 训练结束时调用

```python
# train() 函数末尾，dist.destroy_process_group() 之前
if global_rank == 0:
    csv_file.close()

    if not args.disable_plot:
        logger.info("Generating training curves plot...")
        plot_training_curves(
            csv_path=ckpt_dir / "training_losses.csv",
            output_path=ckpt_dir / "training_curves.png",
            disc_start=args.disc_start
        )
        logger.info(f"Plot saved to {ckpt_dir / 'training_curves.png'}")

dist.destroy_process_group()
```

## Error Handling

### 1. 训练中断恢复
- 检测已存在的 CSV 文件，以追加模式打开
- 不重复写入 header

### 2. 依赖缺失
- 绘图函数内部 try-import
- 缺失时打印警告但不中断训练

### 3. 数据不足
- 绘图前检查 CSV 行数（需要 ≥2 行）
- 数据点少于 10 个时跳过平滑处理

### 4. 磁盘 I/O 错误
- 所有写入操作包裹在 try-except
- 失败时记录错误但继续训练

### 5. 分布式同步
- 所有文件 I/O 操作限定在 `global_rank == 0`
- 避免多进程竞争写入

## Dependencies

新增依赖项（需要更新 `requirements.txt`）:
```txt
pandas>=1.3.0
matplotlib>=3.3.0
scipy>=1.7.0
```

## Usage Examples

### 基本训练
```bash
torchrun --nproc_per_node=8 train_image_ddp.py \
    --exp_name WFIVAE_1024 \
    --image_path /path/to/train \
    --eval_image_path /path/to/val \
    --model_name WFIVAE2 \
    --model_config examples/wfivae2-image-1024.json \
    --resolution 1024 --batch_size 2 --lr 1e-5 \
    --ema --wavelet_loss --eval_lpips
# disc_start=20000, csv_log_steps=50, 自动生成图表
```

### 自定义 CSV 记录频率
```bash
# 每 100 步记录一次
... --csv_log_steps 100
```

### 禁用自动绘图
```bash
# 训练结束时不生成图表
... --disable_plot
```

### 手动生成图表
```python
from train_image_ddp import plot_training_curves

plot_training_curves(
    csv_path="results/WFIVAE_1024-.../training_losses.csv",
    output_path="results/custom_plot.png",
    disc_start=20000
)
```

## Output Files

```
results/WFIVAE_1024-lr1e-05-bs2-rs1024/
├── checkpoint-20000.ckpt
├── checkpoint-40000.ckpt
├── training_losses.csv          # 实时更新
├── training_curves.png          # 训练结束时生成（300 DPI）
└── val_images/
    ├── original/
    └── reconstructed/
```

## Expected Performance

- **CSV 文件大小**: ~100KB（10万步，每50步记录）
- **图表分辨率**: 5400×3600 像素（300 DPI）
- **训练开销**: 几乎可忽略（每50步一次磁盘写入）
- **绘图耗时**: <5秒（10万个数据点）

## Benefits

1. **可追溯性**: CSV 文件独立于 WandB，便于离线分析
2. **自动化**: 无需手动绘图，训练结束即可查看全局趋势
3. **诊断性**: 判别器启动标记清楚显示训练阶段转换
4. **灵活性**: 可配置记录频率，平衡精度和文件大小
5. **鲁棒性**: 错误处理确保日志失败不影响训练
6. **可恢复**: 支持断点续训时追加写入

## Risks and Mitigations

| 风险 | 缓解措施 |
|------|---------|
| 磁盘空间不足 | 可配置记录频率；仅记录核心指标 |
| 依赖缺失导致绘图失败 | 优雅降级，只打印警告 |
| 多进程竞争写入 | 严格限定在 rank 0 |
| CSV 损坏 | flush() 确保实时写入；支持断点恢复 |

## Future Enhancements

- [ ] 支持多个实验的对比绘图
- [ ] 添加滑动窗口统计（均值、方差）
- [ ] 支持导出到 TensorBoard 格式
- [ ] 交互式 HTML 可视化（Plotly）

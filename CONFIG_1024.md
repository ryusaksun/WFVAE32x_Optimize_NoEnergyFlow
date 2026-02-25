# 1024 分辨率训练配置说明（对齐当前代码）

## 1. 推荐启动方式

使用仓库根目录下的 `train_wfivae.sh`：

```bash
cd /Users/ryuichi/Desktop/renxing/WFVAE-series/WF-VAE-8x_disc_start_5

GPU=0,1,2,3,4,5,6,7 \
ORIGINAL_MANIFEST=/mnt/sda/datasets/imagevae_1024/all_manifest.jsonl \
OUTPUT_DIR=/mnt/sdc/WF-VAE-8x_disc_start_5_PatchGANScore_30w \
DISABLE_WANDB=1 \
RESOLUTION=1024 \
EVAL_SUBSET_SIZE=0 \
bash train_wfivae.sh
```

## 2. 1024 推荐参数基线

- `RESOLUTION=1024`
- `BATCH_SIZE=2`（24GB 显存常见起点）
- `EVAL_BATCH_SIZE=2`
- `MAX_STEPS=100000`
- `EVAL_STEPS=1000`
- `SAVE_CKPT_STEP=2000`
- `EVAL_SUBSET_SIZE=0`（验证全量）
- `DATASET_NUM_WORKER=8`

可按显存下调：

```bash
BATCH_SIZE=1 EVAL_BATCH_SIZE=1 DATASET_NUM_WORKER=4 bash train_wfivae.sh
```

## 3. 配置文件说明

当前仓库存在的图像模型配置文件：

- `examples/wfivae2-image-1024.json`

注意：仓库里当前没有 `examples/wfivae2-image.json`。如果你要跑 512 专用结构，请自己提供并通过：

```bash
RESOLUTION=512 MODEL_CONFIG=/path/to/your_512_config.json bash train_wfivae.sh
```

## 4. 与验证相关的关键机制

### 4.1 全量验证开关

`train_image_ddp.py` 当前逻辑：

- `eval_subset_size > 0`：取前 N 个验证样本
- `eval_subset_size <= 0`：使用整个验证集

因此 1024 正式训练建议使用：

```bash
EVAL_SUBSET_SIZE=0
```

### 4.2 PatchGAN patch 分数

每次验证会额外导出：

- 非 EMA：`val_patch_scores/step_xxxxxxxx/`
- EMA：`val_patch_scores/step_xxxxxxxx_ema/`

目录内固定文件：

- `summary.csv`
- `patch_vis/real/*.png`（原图 + patch 热力图 + 叠加图）
- `patch_vis/recon/*.png`（原图 + 重建打分热力图 + 叠加图）

## 5. 曲线与日志

会自动产出：

- `training_losses.csv`
- `training_curves.png`（每次验证后更新）
- `training_curves_final.png`（训练结束）

当前曲线面板：

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

## 6. 训练时长与吞吐建议

经验上 1024 分辨率主要受限于 I/O 与显存：

- 优先保证数据在高速盘
- `DATASET_NUM_WORKER` 从 `4~8` 开始调
- 若验证过慢，先调大 `EVAL_STEPS`

## 7. 常见问题

### 7.1 MAX_STEPS 未停下？

当前版本已修复，达到阈值会打印：

`Reached max_steps=...`

### 7.2 训练不稳定

可尝试：

- 提高 `disc_start`（需要改脚本里固定参数或改为环境变量）
- 降低学习率
- 减小 batch size

### 7.3 文档和脚本不一致

以这三个文件为准：

- `train_wfivae.sh`
- `train_image_ddp.py`
- `AGENTS.md`

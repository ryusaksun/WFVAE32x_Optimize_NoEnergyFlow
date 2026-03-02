# 快速参考卡（当前可用版本）

## 1) 最快启动

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

## 2) 你最常改的参数

```bash
MAX_STEPS=50000 \
EVAL_STEPS=500 \
SAVE_CKPT_STEP=1000 \
BATCH_SIZE=1 \
EVAL_BATCH_SIZE=1 \
CSV_LOG_STEPS=20 \
LOG_STEPS=10 \
DATASET_NUM_WORKER=4 \
EVAL_SUBSET_SIZE=0 \
bash train_wfivae.sh
```

## 3) 关键事实

- 主训练脚本是 `train_wfivae.sh`（不是 `train_image_ddp.sh`）
- 主配置文件是 `examples/wfivae2-image-16chn.json`
- `EVAL_SUBSET_SIZE<=0` 表示验证全量
- `MAX_STEPS` 在当前版本会真实停训

## 4) 训练产物位置

`${OUTPUT_DIR}/${EXP_NAME}-lr...-bs...-rs.../`

- `training_losses.csv`
- `training_curves.png`
- `training_curves_final.png`
- `checkpoint-*.ckpt`
- `val_images/`
- `val_patch_scores/step_xxxxxxxx/`
- `val_patch_scores/step_xxxxxxxx_ema/`

`training_losses.csv` 关键列包含：
`generator_loss`, `discriminator_loss`, `rec_loss`, `perceptual_loss`, `kl_loss`, `wavelet_loss`,
`psnr`, `lpips`, `psnr_ema`, `lpips_ema`

## 5) PatchGAN 分数文件

每个 `val_patch_scores` 目录包含：

- `summary.csv`
- `patch_vis/real/*.png`（原图 + patch 热力图 + 叠加图）
- `patch_vis/recon/*.png`（原图 + 重建打分热力图 + 叠加图）

## 6) 常见排查

### OOM

- `BATCH_SIZE=1`
- `EVAL_BATCH_SIZE=1`
- `DATASET_NUM_WORKER=2`

### 验证太慢

- 增大 `EVAL_STEPS`
- 临时设置 `EVAL_SUBSET_SIZE=200`（调试时）

### 恢复训练

```bash
RESUME_CKPT=/path/to/checkpoint-20000.ckpt bash train_wfivae.sh
```

## 7) 最小检查命令

```bash
bash -n train_wfivae.sh
python3 -m py_compile train_image_ddp.py
```

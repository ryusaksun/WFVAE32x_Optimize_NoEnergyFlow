# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WF-VAE (Wavelet-driven energy Flow VAE) is an Image VAE architecture using multi-level Haar wavelet transforms to construct efficient energy flow pathways. Accepted by CVPR 2025. The model achieves 8× spatial compression (2× wavelet + 4× downsampling).

## Common Commands

### Environment Setup
```bash
conda create -n wfvae python=3.10 -y
conda activate wfvae
pip install -r requirements.txt
```

### Training
```bash
# Recommended: use the launch script (handles dataset splitting, DDP, cleanup)
bash train_wfivae.sh

# Multi-GPU with env vars
GPU=0,1,2,3 bash train_wfivae.sh

# Resume from checkpoint
RESUME_CKPT="/path/to/checkpoint.ckpt" bash train_wfivae.sh

# Override common parameters
RESOLUTION=512 BATCH_SIZE=8 MAX_STEPS=50000 EVAL_STEPS=500 bash train_wfivae.sh

# Direct torchrun (low-level, requires manual dataset setup)
torchrun --nproc_per_node=8 train_image_ddp.py \
    --exp_name WFIVAE_1024 \
    --image_path /path/to/train_images \
    --eval_image_path /path/to/eval_images \
    --model_name WFIVAE2 \
    --model_config examples/wfivae2-image-1024.json \
    --resolution 1024 --batch_size 2 --lr 1e-5 \
    --ema --wavelet_loss --eval_lpips
```

### Testing
```bash
python scripts/test_imagevae.py
```

### Pre-commit Validation
```bash
python3 -m py_compile train_image_ddp.py
bash -n train_wfivae.sh
```
If you changed dataset or validation output, run a minimal training to the first validation to confirm `val_patch_scores/`, `training_curves.png` generate correctly.

### Inference
```bash
python scripts/recon_single_image.py \
    --model_name WFIVAE2 \
    --from_pretrained /path/to/model \
    --image_path input.jpg --rec_path output.jpg
```

### Other Scripts
- `scripts/save_hf_model.py` — Convert checkpoint to HuggingFace format
- `scripts/merge_encoder_and_decoder.py` — Merge separately trained encoder/decoder

## Architecture

### Wavelet Energy Flow Pipeline

The core innovation is explicit frequency-domain information flow via Haar wavelets:

- **Encoder**: Input → `HaarWaveletTransform2D` (RGB → 12 coefficients: 4 sub-bands × 3 channels, at half resolution) → `WFDownBlock` (merge wavelet energy flow with spatial features via concat, then downsample) × 2 → mid blocks → latent distribution
- **Decoder**: Latent → mid blocks → `WFUpBlock` (split energy flow, generate wavelet coefficients via outflow, upsample) × 2 → `InverseHaarWaveletTransform2D` → output

Key: the decoder is fully independent — no skip connections from encoder. Energy flow is reconstructed entirely from the latent.

### Training Loop Design

`train_image_ddp.py` uses **alternating optimization**: odd steps optimize the discriminator, even steps optimize the generator. The discriminator activates at `--disc_start` (default 80000, this repo uses 5). Discriminator weight is computed adaptively based on gradient norms of the last decoder layer.

**Combined loss** (generator step):
```
generator_loss = rec_loss/exp(logvar) + logvar + perceptual_loss + kl_loss + disc_loss + wavelet_loss
```
where `disc_loss = -mean(D(recon)) * adaptive_weight * disc_factor` and `wavelet_loss = l1(encoder_coeffs, decoder_coeffs_reversed)`.

### Checkpoint Format

Checkpoints use keys: `gen_model`, `dics_model` (note: historical typo for discriminator — do NOT rename, it would break checkpoint resumption), `optimizer_state`, `ema_state_dict`, `scaler_state`, `sampler_state`. When loading for inference, EMA weights are preferred over normal state_dict.

### Registry Pattern

Models register via decorator and load dynamically:
```python
model_cls = ModelRegistry.get_model("WFIVAE2")
model = model_cls.from_pretrained("/path/to/checkpoint")
# or from config
model = model_cls.from_config("examples/wfivae2-image-1024.json")
```

### Key Source Files

| Area | File |
|------|------|
| Model | `causalimagevae/model/vae/modeling_wfvae2.py` |
| Wavelet | `causalimagevae/model/modules/wavelet.py` |
| Loss/Discriminator | `causalimagevae/model/losses/perceptual_loss_2d.py` |
| PatchGAN | `causalimagevae/model/losses/discriminator.py` |
| EMA | `causalimagevae/model/ema_model.py` |
| Registry | `causalimagevae/model/registry.py` |
| DDP Sampler | `causalimagevae/dataset/ddp_sampler.py` |
| Dataset | `causalimagevae/dataset/image_dataset.py`, `manifest_dataset.py` |

## Key Training Parameters

| Parameter | Value |
|-----------|-------|
| `--model_name` | WFIVAE2 |
| `--model_config` | wfivae2-image-1024.json |
| `--disc_cls` | causalimagevae.model.losses.LPIPSWithDiscriminator |
| `--disc_start` | 5 (this repo), 80000 (default) |
| `--kl_weight` | 1e-6 |
| `--wavelet_weight` | 0.1 |
| `--disc_weight` | 0.5 |
| `--perceptual_weight` | 1.0 |
| `--loss_type` | l1 |

### Loss Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv_log_steps` | 50 | Log losses to CSV every N steps |
| `--disable_plot` | False | Disable automatic plot generation |

CSV fields: `step, generator_loss, discriminator_loss, rec_loss, kl_loss, wavelet_loss, psnr, lpips`. Output to `{ckpt_dir}/training_losses.csv`.

Plot files: `training_curves.png` (at checkpoints), `training_curves_final.png` (normal completion), `training_curves_interrupted.png` (Ctrl+C). Features 3×3 subplot layout with smoothed curves and disc_start marker.

### Training Output Structure

Under `{ckpt_dir}/{exp_name-lr...-bs...-rs...}/`:
- `training_losses.csv` — metrics CSV
- `training_curves*.png` — loss plots
- `val_images/original/` and `val_images/reconstructed/` — validation images
- `val_patch_scores/step_xxxxxxxx/` and `step_xxxxxxxx_ema/` — PatchGAN scores with `summary.csv` + `patch_vis/{real,recon}/*.png` heatmaps
- `checkpoint-*.ckpt` — model + optimizer state

## Model Configs

Located in `examples/`:
- `wfivae2-image-1024.json` — 1024px, latent_dim=8, base_channels=[128,256,512]

### Latent Dimensions
- 1024px input → latent shape `[B, 8, 128, 128]`
- 512px input → latent shape `[B, 8, 64, 64]`
- Spatial compression: 8× (2× wavelet + 4× from 2 downsampling blocks)

## Data Format

Images organized recursively in directories (jpg/png/webp/bmp). File list is cached to pickle for faster subsequent loads. Optionally use JSONL manifest files with `--use_manifest`:
```json
{"image_path": "/path/to/image.jpg"}
```
Manifest supports field name aliases: `image_path`, `path`, or `target`.

## Distributed Training Notes

- PyTorch DDP with NCCL backend; `train_image_ddp.py` always requires DDP environment (use torchrun even for single-GPU)
- Set `CUDA_VISIBLE_DEVICES` before `torchrun`
- Checkpoints and WandB logging on rank 0 only
- Validation results gathered across all ranks via `dist.all_reduce` and `all_gather_object`

## High-Risk Modification Points

- DDP gather logic in `train_image_ddp.py` uses `all_gather_object` — changing validation return structure requires updating the gather/logging/save chain together
- Patch score export depends on `index` field in validation batches; both `ValidImageDataset` and `ValidManifestImageDataset` provide it — don't drop it when modifying datasets
- Checkpoint discriminator key is `dics_model` (typo) — renaming breaks old checkpoint loading
- Do not modify data paths to your local private paths without confirmation

## Documentation Sync

When upgrading scripts or workflows, keep these docs in sync:
- `README_image_training.md`
- `QUICK_REFERENCE.md`
- `CONFIG_1024.md`

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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 train_image_ddp.py \
    --exp_name WFIVAE_1024 \
    --image_path /path/to/train_images \
    --eval_image_path /path/to/eval_images \
    --model_name WFIVAE2 \
    --model_config examples/wfivae2-image-1024.json \
    --resolution 1024 --batch_size 2 --lr 1e-5 \
    --ema --wavelet_loss --eval_lpips
```

See `train_wfivae.sh` for a full launch script with manifest support and checkpoint resumption.

### Testing
```bash
python scripts/test_imagevae.py
```
Tests model creation, forward pass at multiple resolutions, dimension verification, wavelet coefficient extraction, encoder/decoder separation, and EMA state management.

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

### Training Loop Design

`train_image_ddp.py` uses **alternating optimization**: odd steps optimize the discriminator, even steps optimize the generator. The discriminator activates at `--disc_start` (default 80000). Discriminator weight is computed adaptively based on gradient norms of the last decoder layer.

### Checkpoint Format

Checkpoints use keys: `gen_model`, `dics_model` (note: historical typo for discriminator), `optimizer_state`, `ema_state_dict`, `scaler_state`, `sampler_state`. When loading for inference, EMA weights are preferred over normal state_dict.

### Registry Pattern

Models register via decorator and load dynamically:
```python
model_cls = ModelRegistry.get_model("WFIVAE2")
model = model_cls.from_pretrained("/path/to/checkpoint")
# or from config
model = model_cls.from_config("examples/wfivae2-image-1024.json")
```

## Key Training Parameters

| Parameter | Value |
|-----------|-------|
| `--model_name` | WFIVAE2 |
| `--model_config` | wfivae2-image-1024.json |
| `--disc_cls` | causalimagevae.model.losses.LPIPSWithDiscriminator |
| `--disc_start` | 80000 |
| `--kl_weight` | 1e-6 |
| `--wavelet_weight` | 0.1 |

### Loss Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv_log_steps` | 50 | Log losses to CSV every N steps |
| `--disable_plot` | False | Disable automatic plot generation |

CSV fields: `step, generator_loss, discriminator_loss, rec_loss, kl_loss, wavelet_loss, psnr, lpips`. Output to `{ckpt_dir}/training_losses.csv`.

Plot files: `training_curves.png` (at checkpoints), `training_curves_final.png` (normal completion), `training_curves_interrupted.png` (Ctrl+C). Features 3×3 subplot layout with smoothed curves and disc_start marker.

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

- PyTorch DDP with NCCL backend
- Set `CUDA_VISIBLE_DEVICES` before `torchrun`
- Checkpoints and WandB logging on rank 0 only
- Validation results gathered across all ranks via `dist.all_reduce`

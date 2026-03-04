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
RESOLUTION=512 BATCH_SIZE=8 EPOCHS=500 EVAL_STEPS=500 bash train_wfivae.sh

# Direct torchrun (low-level, requires manual dataset setup)
torchrun --nproc_per_node=8 train_image_ddp.py \
    --exp_name WFIVAE_1024 \
    --image_path /path/to/train_images \
    --eval_image_path /path/to/eval_images \
    --model_name WFIVAE2 \
    --model_config examples/wfivae2-image-192bc.json \
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
If you changed dataset or validation output, run a minimal training to the first validation to confirm `training_curves.png` generates correctly.

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

### Channel Dimension Flow (192bc config, 512px input)

**Encoder** (spatial: 512→256→128→64):
```
[3, 512]  →Wavelet→  [12, 256]  →conv_in→  [192, 256]
  →WFDownBlock1: ResBlock(192) → Downsample → concat [192+128 flow] → ResBlock(320→384)  → [384, 128]
  →WFDownBlock2: ResBlock(384) → Downsample → concat [384+128 flow] → ResBlock(512→768)  → [768, 64]
  →Mid(768) → norm → conv_out(768→64)  → [64, 64]   (64 = 2×latent_dim: mean+var)
```

**Decoder** (spatial: 64→128→256→512):
```
[32, 64]  →conv_in→  [768, 64]  →Mid(768)
  →WFUpBlock1: branch(768→896) → split [768 main | 128 flow→12→InvWavelet→w] → ResBlock→Up→ResBlock(768→384) → [384, 128]
  →WFUpBlock2: branch(384→512) → split [384 main | 128 flow→12→InvWavelet→w] → ResBlock→Up→ResBlock(384→192) → [192, 256]
  →norm → conv_out(192→12) → merge w → InvWavelet → [3, 512]
```

WFDownBlock energy flow: wavelet coefficients (12ch) → `in_flow_conv` (12→`energy_flow_size`) → concat with main trunk.
WFUpBlock energy flow: split `energy_flow_size` channels from branch output → `out_flow_conv` → 12ch wavelet coefficients → InverseWavelet → RGB residual `w`.

### Training Loop Design

`train_image_ddp.py` uses **alternating optimization** via `current_step % 2`: even steps (0, 2, 4…) optimize the generator, odd steps (1, 3, 5…) optimize the discriminator. The discriminator only activates when `current_step >= disc_start` (default 80000, this repo uses 5). Discriminator weight is computed adaptively based on gradient norms of the last decoder layer.

**Combined loss** (generator step):
```
generator_loss = rec_loss/exp(logvar) + logvar + perceptual_loss + kl_loss + disc_loss + wavelet_loss
```
where `disc_loss = -mean(D(recon)) * adaptive_weight * disc_factor` and `wavelet_loss = l1(encoder_coeffs, decoder_coeffs_reversed)`.

**TTUR**: Discriminator can use a separate learning rate (`--disc_lr`, default same as generator). Two Time-scale Update Rule recommends 2-4× the generator LR for faster discriminator convergence.

**Learned logvar** (`--learn_logvar`): When enabled, `logvar` becomes a trainable scalar with its own optimizer (lr controlled by `--logvar_lr`, default 1e-2), auto-balancing reconstruction loss vs GAN loss magnitude.

Mixed precision: default bfloat16 (`--mix_precision bf16`), also supports `fp16` and `fp32`. Uses `torch.amp.GradScaler`.

Signal handling: Ctrl+C triggers graceful shutdown — saves `training_curves_interrupted.png` and exits cleanly on rank 0. Image normalization: input/output in [-1, 1] range (torchvision `Normalize([0.5]*3, [0.5]*3)`).

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
| Discriminator | `causalimagevae/model/losses/discriminator.py` |
| EMA | `causalimagevae/model/ema_model.py` |
| Registry | `causalimagevae/model/registry.py` |
| DDP Sampler | `causalimagevae/dataset/ddp_sampler.py` |
| Dataset | `causalimagevae/dataset/image_dataset.py`, `manifest_dataset.py` |

## Key Training Parameters

| Parameter | Value |
|-----------|-------|
| `--model_name` | WFIVAE2 |
| `--model_config` | wfivae2-image-192bc.json (default), wfivae2-image-16chn.json (smaller), wfivae2-image-1024.json (8chn legacy) |
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
| `--disable_wandb` | False | Disable WandB logging (wandb is optional import) |
| `--eval_subset_size` | 30 | Validation subset size (0 = full set) |
| `--eval_num_image_log` | 20 | Number of validation images to save and visualize |

CSV fields: `step, generator_loss, discriminator_loss, rec_loss, perceptual_loss, kl_loss, wavelet_loss, nll_loss, g_loss, d_weight, logits_real, logits_fake, nll_grads_norm, g_grads_norm, psnr, lpips, psnr_ema, lpips_ema`. Output to `{ckpt_dir}/{exp_name}.csv`.

Plot files: `training_curves.png` (at checkpoints), `training_curves_final.png` (normal completion), `training_curves_interrupted.png` (Ctrl+C). Features 7×3 subplot layout with smoothed curves and disc_start marker.

### Training Output Structure

Under `{ckpt_dir}/{exp_name-lr...-bs...-rs...}/`:
- `{exp_name}.csv` — metrics CSV
- `training_curves*.png` — loss plots
- `val_images/original/` and `val_images/reconstructed/` — validation images
- `checkpoint-*.ckpt` — model + optimizer state

## Model Configs

Located in `examples/`:
- `wfivae2-image-192bc.json` — latent_dim=32, base_channels=[192,384,768] (default, aligned with official WF-VAE large config)
- `wfivae2-image-16chn.json` — latent_dim=16, base_channels=[128,256,512] (smaller variant)
- `wfivae2-image-1024.json` — latent_dim=8, base_channels=[128,256,512] (legacy)

### Latent Dimensions
- 1024px input → latent shape `[B, 32, 128, 128]` (192bc config) / `[B, 16, 128, 128]` (16chn config)
- 512px input → latent shape `[B, 32, 64, 64]` / `[B, 16, 64, 64]`
- 256px input → latent shape `[B, 32, 32, 32]` / `[B, 16, 32, 32]`
- Spatial compression: 8× (2× wavelet + 4× from 2 downsampling blocks)

## Data Format

Images organized recursively in directories (jpg/png/webp/bmp). File list is cached to pickle for faster subsequent loads. Optionally use JSONL manifest files with `--use_manifest`:
```json
{"image_path": "/path/to/image.jpg"}
```
Manifest supports field name aliases: `image_path`, `path`, or `target`.

## `train_wfivae.sh` Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU` | `6` (check script) | GPU indices, e.g. `0,1,2,3` for multi-GPU DDP |
| `RESOLUTION` | `256` | Image resolution (256, 512, or 1024) |
| `BATCH_SIZE` | `16` (256px) / `8` (512px) / `2` (1024px) | Per-GPU batch size |
| `EPOCHS` | `1000` | Training epochs |
| `EVAL_STEPS` | `1000` | Validation interval |
| `SAVE_CKPT_STEP` | `2000` | Checkpoint save interval |
| `EVAL_SUBSET_SIZE` | `30` | Validation subset (0 = full set) |
| `EVAL_NUM_IMAGE_LOG` | `20` | Number of val images to save & visualize |
| `CSV_LOG_STEPS` | `50` | CSV logging frequency |
| `LOG_STEPS` | `10` | Console logging frequency |
| `DATASET_NUM_WORKER` | `8` | DataLoader workers |
| `RESUME_CKPT` | — | Path to checkpoint for resumption |
| `DISC_LR` | `1e-5` | Discriminator learning rate (TTUR: set 2-4× gen LR) |
| `LEARN_LOGVAR` | — | Non-empty to enable learned logvar |
| `LOGVAR_LR` | `1e-2` | Learning rate for logvar parameter |
| `ORIGINAL_MANIFEST` | `/mnt/sdb/kinetics400_frames/train_manifest.jsonl` | Training JSONL manifest path |
| `VAL_MANIFEST` | `/mnt/sdb/kinetics400_frames/val_manifest.jsonl` | Pre-split validation manifest (skips auto-split) |
| `OUTPUT_DIR` | `/mnt/sdb/{project_name}` | Output directory |
| `DISABLE_WANDB` | `1` | `1`/`true`/`yes` to disable WandB |
| `TRAIN_RATIO` | `0.9` | Train/val split ratio (only when no VAL_MANIFEST) |
| `MIX_PRECISION` | `bf16` | Training precision: `bf16`, `fp16`, or `fp32` |
| `ADAPTIVE_WEIGHT_CLAMP` | `1e6` | Max adaptive weight for discriminator |

## Distributed Training Notes

- PyTorch DDP with NCCL backend; `train_image_ddp.py` always requires DDP environment (use torchrun even for single-GPU)
- Set `CUDA_VISIBLE_DEVICES` before `torchrun`
- Checkpoints and WandB logging on rank 0 only
- Validation results gathered across all ranks via `dist.all_reduce` and `all_gather_object`

## High-Risk Modification Points

- DDP gather logic in `train_image_ddp.py` uses `all_gather_object` — changing validation return structure requires updating the gather/logging/save chain together
- Checkpoint discriminator key is `dics_model` (typo) — renaming breaks old checkpoint loading
- Do not modify data paths to your local private paths without confirmation

## Documentation Sync

When upgrading scripts or workflows, keep these docs in sync:
- `README_image_training.md`
- `QUICK_REFERENCE.md`

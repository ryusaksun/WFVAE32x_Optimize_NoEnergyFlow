# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WF-VAE (Wavelet-driven energy Flow VAE) is an Image VAE architecture using multi-level Haar wavelet transforms to construct efficient energy flow pathways. Accepted by CVPR 2025. Spatial compression ratio is determined by `base_channels` length: `compression = 2^(len(base_channels))`. The default 8x config uses 3 elements (2x wavelet + 4x from 2 downsampling blocks); the 32x config uses 5 elements (2x wavelet + 16x from 4 downsampling blocks). Fully convolutional — no positional encodings, no fixed spatial dimensions, so weights are resolution-agnostic.

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

# Gradient accumulation (effective batch = BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS)
GRAD_ACCUM_STEPS=4 BATCH_SIZE=1 RESOLUTION=1024 bash train_wfivae.sh

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
# Default 8x config
python scripts/test_imagevae.py

# 32x config
python scripts/test_imagevae.py --config examples/wfivae2-image-32x-192bc.json
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
- `scripts/recon_single_image_flux.py` — FLUX.2 VAE reconstruction (for comparison with WFIVAE2)

## Architecture

### Wavelet Energy Flow Pipeline

The core innovation is explicit frequency-domain information flow via Haar wavelets:

- **Encoder**: Input -> `HaarWaveletTransform2D` (RGB -> 12 coefficients: 4 sub-bands x 3 channels, at half resolution) -> `WFDownBlock` (merge wavelet energy flow with spatial features via concat, then downsample) x N -> mid blocks -> latent distribution
- **Decoder**: Latent -> mid blocks -> `WFUpBlock` (split energy flow, generate wavelet coefficients via outflow, upsample) x N -> `InverseHaarWaveletTransform2D` -> output

Key: the decoder is fully independent — no skip connections from encoder. Energy flow is reconstructed entirely from the latent.

**DCAE-style residual shortcuts**: Both WFDownBlock and WFUpBlock include non-parametric residual connections inspired by DC-AE. `WFDownBlock._down_shortcut` uses `pixel_unshuffle(x, 2)` (space-to-channel) + channel averaging to match output dimensions. `WFUpBlock._up_shortcut` uses `pixel_shuffle(x, 2)` (channel-to-space) + channel duplication. These shortcuts bypass the main trunk (ResBlocks + wavelet flow), improving gradient flow in deep 32x configs.

**Asymmetric encoder/decoder**: Encoder and decoder can have different `num_resblocks` and `base_channels`. The 32x config uses `encoder_num_resblocks=[2,3,9,2], decoder_num_resblocks=[3,4,12,2]` — the decoder is deeper per-stage (especially at Up2) to compensate for the lack of skip connections. Total params ≈ 478 M (enc 169 M / dec 309 M).

### Channel Dimension Flow (64chn-192bc config, 512px input)

**Encoder** (spatial: 512->256->128->64):
```
[3, 512]  ->Wavelet->  [12, 256]  ->conv_in->  [192, 256]
  ->WFDownBlock1: ResBlock(192) -> Downsample -> concat [192+128 flow] -> ResBlock(320->384)  -> [384, 128]
  ->WFDownBlock2: ResBlock(384) -> Downsample -> concat [384+128 flow] -> ResBlock(512->768)  -> [768, 64]
  ->Mid(768) -> norm -> conv_out(768->128)  -> [128, 64]   (128 = 2*latent_dim: mean+var)
```

**Decoder** (spatial: 64->128->256->512):
```
[64, 64]  ->conv_in->  [768, 64]  ->Mid(768)
  ->WFUpBlock1: branch(768->896) -> split [768 main | 128 flow->12->InvWavelet->w] -> ResBlock->Up->ResBlock(768->384) -> [384, 128]
  ->WFUpBlock2: branch(384->512) -> split [384 main | 128 flow->12->InvWavelet->w] -> ResBlock->Up->ResBlock(384->192) -> [192, 256]
  ->norm -> conv_out(192->12) -> merge w -> InvWavelet -> [3, 512]
```

WFDownBlock energy flow: wavelet coefficients (12ch) -> `in_flow_conv` (12->`energy_flow_size`) -> concat with main trunk.
WFUpBlock energy flow: split `energy_flow_size` channels from branch output -> `out_flow_conv` -> 12ch wavelet coefficients -> InverseWavelet -> RGB residual `w`.

### Training Loop Design

`train_image_ddp.py` uses **alternating optimization** via `current_step % 2`: even steps (0, 2, 4...) optimize the generator, odd steps (1, 3, 5...) optimize the discriminator. The discriminator only activates when `current_step >= disc_start` (default 80000, this repo uses 5). Discriminator weight is computed adaptively based on gradient norms of the last decoder layer.

**Combined loss** (generator step):
```
generator_loss = rec_loss/exp(logvar) + logvar + perceptual_loss + kl_loss + disc_loss + wavelet_loss
```
where `disc_loss = -mean(D(recon)) * adaptive_weight * disc_factor` and `wavelet_loss = l1(encoder_coeffs, decoder_coeffs_reversed)`.

**Gradient accumulation** (`--gradient_accumulation_steps N`): Accumulates gradients over N micro-batches before each optimizer step. Uses `model.no_sync()` / `disc.no_sync()` on non-final micro-batches to skip DDP all-reduce. Loss is divided by N. The step type (gen/disc) is determined at the start of each accumulation cycle and held constant across micro-batches. Partial accumulation at epoch boundaries is discarded.

**TTUR**: Discriminator can use a separate learning rate (`--disc_lr`, default same as `--lr`). Two Time-scale Update Rule recommends 2-4x the generator LR for faster discriminator convergence.

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
| `--model_config` | wfivae2-image-32x-192bc.json (**default in this 32x repo**), wfivae2-image-64chn-192bc.json (64chn 8x), wfivae2-image-192bc.json (32chn 8x), wfivae2-image-16chn.json (smaller 8x), wfivae2-image-1024.json (8chn legacy) |
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

Plot files: `training_curves.png` (at checkpoints), `training_curves_final.png` (normal completion), `training_curves_interrupted.png` (Ctrl+C). Features 7x3 subplot layout with smoothed curves and disc_start marker.

### Training Output Structure

Under `{ckpt_dir}/{exp_name-lr...-bs...-rs...}/`:
- `{exp_name}.csv` — metrics CSV
- `training_curves*.png` — loss plots
- `val_images/original/` and `val_images/reconstructed/` — validation images
- `checkpoint-*.ckpt` — model + optimizer state

## Model Configs

Located in `examples/`. Compression ratio is `2^(len(base_channels))`:
- `wfivae2-image-32x-192bc.json` — latent_dim=64, base_channels=[256,512,512,1024,1024], enc_resblocks=[2,3,9,2], dec_resblocks=[3,4,12,2], **32x** (4 down/up blocks, DCAE-style progressive channels and per-stage depth — heavy middle at Down3/Up3 with **middle-body depth 8/10 matching DC-AE f32c32's Stage 2**, thin edges at Down1/Down4/Up1/Up4, **default in this repo**). Note: `num_resblocks=N` counts the full main-trunk ResBlocks = 1 branch_conv (decoder only) + middle res_block seq + 1 out_res_block, so WFDownBlock middle depth = N−1 and WFUpBlock middle depth = N−2 (vs. DC-AE `depth_list` which is the pure main-body count). `num_resblocks` accepts either a single int (legacy, uniform) or a list of length `len(base_channels)-1` indexed from the highest-res stage to the lowest-res.
- `wfivae2-image-64chn-192bc.json` — latent_dim=64, base_channels=[192,384,768], enc_resblocks=2, dec_resblocks=3, 8x (high-capacity latent)
- `wfivae2-image-192bc.json` — latent_dim=32, base_channels=[192,384,768], 8x (aligned with official WF-VAE large config)
- `wfivae2-image-16chn.json` — latent_dim=16, base_channels=[128,256,512], 8x (smaller variant)
- `wfivae2-image-1024.json` — latent_dim=8, base_channels=[128,256,512], 8x (legacy)

### Latent Dimensions (8x configs)
- 1024px input -> latent shape `[B, 64, 128, 128]` (64chn-192bc config) / `[B, 32, 128, 128]` (192bc config) / `[B, 16, 128, 128]` (16chn config)
- 512px input -> latent shape `[B, 64, 64, 64]` / `[B, 32, 64, 64]` / `[B, 16, 64, 64]`
- 256px input -> latent shape `[B, 64, 32, 32]` / `[B, 32, 32, 32]` / `[B, 16, 32, 32]`
- Spatial compression: 8x (2x wavelet + 4x from 2 downsampling blocks)

### Latent Dimensions (32x config)
- 1024px input -> latent shape `[B, 64, 32, 32]`
- 512px input -> latent shape `[B, 64, 16, 16]`
- 256px input -> latent shape `[B, 64, 8, 8]`
- Spatial compression: 32x (2x wavelet + 16x from 4 downsampling blocks)

## Data Format

Images organized recursively in directories (jpg/png/webp/bmp). File list is cached to pickle for faster subsequent loads. Optionally use JSONL manifest files with `--use_manifest`:
```json
{"image_path": "/path/to/image.jpg"}
```
Manifest supports field name aliases: `image_path`, `path`, or `target`.

## `train_wfivae.sh` Environment Variables

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU` | `0,1` | GPU indices, e.g. `0,1,2,3` for multi-GPU DDP |
| `RESOLUTION` | `256` | Image resolution (256, 512, or 1024) |
| `BATCH_SIZE` | `8` (256px) / `8` (512px) / `2` (1024px) | Per-GPU batch size |
| `EPOCHS` | `1000` | Training epochs |
| `EVAL_STEPS` | `1000` | Validation interval |
| `SAVE_CKPT_STEP` | `5000` | Checkpoint save interval |
| `EVAL_SUBSET_SIZE` | `30` | Validation subset (0 = full set) |
| `EVAL_NUM_IMAGE_LOG` | `20` | Number of val images to save & visualize |
| `CSV_LOG_STEPS` | `50` | CSV logging frequency |
| `LOG_STEPS` | `10` | Console logging frequency |
| `DATASET_NUM_WORKER` | `8` | DataLoader workers |
| `EXP_NAME` | `{PROJECT_NAME}` | Experiment name (used in output dir and CSV filename) |
| `MODEL_CONFIG` | `wfivae2-image-32x-192bc.json` | Model config path (auto-selected per resolution in this repo) |
| `MAX_STEPS` | — | Stop after N steps (overrides `EPOCHS` if set) |
| `EVAL_BATCH_SIZE` | 2 (1024px) / 4 (512px) / 8 (256px) | Validation batch size |
| `WANDB_PROJECT` | `WFIVAE` | WandB project name |
| `MASTER_PORT` | auto-detect 29500-29599 | torchrun master port |

### Learning Rate & Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `LR` | `1e-5` | Generator learning rate |
| `DISC_LR` | same as `LR` | Discriminator learning rate (TTUR: set higher than LR) |
| `GRAD_ACCUM_STEPS` | `1` | Gradient accumulation steps (effective batch = BATCH_SIZE * GPUs * this) |
| `LEARN_LOGVAR` | — | Non-empty to enable learned logvar |
| `LOGVAR_LR` | `1e-2` | Learning rate for logvar parameter |
| `ADAPTIVE_WEIGHT_CLAMP` | `1e6` | Max adaptive weight for discriminator |
| `MIX_PRECISION` | `bf16` | Training precision: `bf16`, `fp16`, or `fp32` |

### Resume

| Variable | Default | Description |
|----------|---------|-------------|
| `RESUME_CKPT` | — | Path to checkpoint for resumption |

### Data Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `ORIGINAL_MANIFEST` | SA-1B_256 or SA-1B_512 (resolution-dependent, on `/mnt/hpfs/HDU/ssk/`) | Training JSONL manifest path |
| `VAL_MANIFEST` | SA-1B_256 or SA-1B_512 val manifest (resolution-dependent) | Pre-split validation manifest (skips auto-split) |
| `OUTPUT_DIR` | `/mnt/hpfs/HDU/ssk/Exp_Output/{project_name}` | Output directory |
| `DISABLE_WANDB` | `1` | `1`/`true`/`yes` to disable WandB |
| `TRAIN_RATIO` | `0.9` | Train/val split ratio (only when no VAL_MANIFEST) |

## Distributed Training Notes

- PyTorch DDP with NCCL backend; `train_image_ddp.py` always requires DDP environment (use torchrun even for single-GPU)
- `--find_unused_parameters` is enabled by default (needed because gen/disc alternation means some params have no grad on each step)
- Set `CUDA_VISIBLE_DEVICES` before `torchrun`
- Checkpoints and WandB logging on rank 0 only
- Validation results gathered across all ranks via `dist.all_reduce` and `all_gather_object`
- Gradient accumulation uses `model.no_sync()` to skip all-reduce on non-final micro-batches

## High-Risk Modification Points

- DDP gather logic in `train_image_ddp.py` uses `all_gather_object` — changing validation return structure requires updating the gather/logging/save chain together
- Checkpoint discriminator key is `dics_model` (typo) — renaming breaks old checkpoint loading
- Do not modify data paths to your local private paths without confirmation
- Gradient accumulation interacts with gen/disc alternation: step type is locked for the entire accumulation cycle, so each logical step still alternates correctly
- `ValidImageDataset` and `ValidManifestImageDataset` both provide an `index` field for deduplication during validation gather — do not drop it when modifying dataset classes
- When modifying generator loss path, `learn_logvar`, or DDP wrapper boundaries, also check `aux_gen_params` sync logic (manual gradient sync for non-DDP-wrapped params)
- `sampler_state` in checkpoint stores "consumed batch position" — modifying checkpoint save/resume must verify epoch boundary recovery doesn't replay samples

## Documentation Sync

When upgrading scripts or workflows, keep these docs in sync:
- `docs/README_image_training.md` (note: currently references outdated features like `val_patch_scores` and `training_losses.csv` — the CSV is now `{exp_name}.csv`)
- `docs/QUICK_REFERENCE.md` (note: references outdated paths and old config defaults)

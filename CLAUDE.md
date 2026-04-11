# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WF-VAE (Wavelet-driven energy Flow VAE) is an Image VAE architecture using multi-level Haar wavelet transforms to construct efficient energy flow pathways. Accepted by CVPR 2025. Spatial compression ratio is determined by `base_channels` length: `compression = 2^(len(base_channels))`. **This branch ships only one config (`wfivae2-image-32x-192bc.json`, 32x compression, `base_channels=[256,512,512,1024,1024]`, `latent_dim=64`)** — all legacy 8x configs have been removed. The model is fully convolutional — no positional encodings, no fixed spatial dimensions, so weights are resolution-agnostic.

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

# Override common parameters (RESOLUTION only supports 256 or 1024)
RESOLUTION=1024 BATCH_SIZE=2 EPOCHS=500 EVAL_STEPS=500 bash train_wfivae.sh

# Gradient accumulation (effective batch = BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS)
GRAD_ACCUM_STEPS=4 BATCH_SIZE=1 RESOLUTION=1024 bash train_wfivae.sh

# Switch discriminator mode (default is multiscale + spectral_norm)
DISC_TYPE=single DISC_NORM=bn bash train_wfivae.sh   # legacy 1-scale PatchGAN + BN
FEAT_MATCH_WEIGHT=0 bash train_wfivae.sh             # multiscale but disable FM loss

# Direct torchrun (low-level, requires manual dataset setup)
torchrun --nproc_per_node=8 train_image_ddp.py \
    --exp_name WFIVAE_1024 \
    --image_path /path/to/train_images \
    --eval_image_path /path/to/eval_images \
    --model_name WFIVAE2 \
    --model_config examples/wfivae2-image-32x-192bc.json \
    --resolution 1024 --batch_size 2 --lr 1e-5 \
    --ema --wavelet_loss --eval_lpips
```

### Testing
```bash
# Runs forward/backward sanity check on the 32x config at 256px/512px/1024px
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
- `scripts/draw_architecture.py` — Matplotlib architecture diagram generator (outputs `wfvae2_32x_dcae_architecture.png` + `wfvae2_32x_dcae_blocks_detail.png`)
- `architecture_diagram.html` — Interactive mermaid-based architecture reference (overview, encoder, decoder, WFDownBlock, WFUpBlock, wavelet flow, discriminator & loss pipeline). Open directly in browser.
- `scripts/resize_sa1b_256.py` / `resize_sa1b_512.py` / `resize_faces_to_256_512.py` — Dataset preprocessing tools

## Architecture

### Wavelet Energy Flow Pipeline

The core innovation is explicit frequency-domain information flow via Haar wavelets:

- **Encoder**: Input -> `HaarWaveletTransform2D` (RGB -> 12 coefficients: 4 sub-bands x 3 channels, at half resolution) -> `WFDownBlock` (merge wavelet energy flow with spatial features via concat, then downsample) x N -> mid blocks -> latent distribution
- **Decoder**: Latent -> mid blocks -> `WFUpBlock` (split energy flow, generate wavelet coefficients via outflow, upsample) x N -> `InverseHaarWaveletTransform2D` -> output

Key: the decoder is fully independent — no skip connections from encoder. Energy flow is reconstructed entirely from the latent.

**DCAE-style residual shortcuts**: Both WFDownBlock and WFUpBlock include non-parametric residual connections inspired by DC-AE. `WFDownBlock._down_shortcut` uses `pixel_unshuffle(x, 2)` (space-to-channel) + channel averaging to match output dimensions. `WFUpBlock._up_shortcut` uses `pixel_shuffle(x, 2)` (channel-to-space) + channel duplication. These shortcuts bypass the main trunk (ResBlocks + wavelet flow), improving gradient flow in deep 32x configs.

**Asymmetric encoder/decoder**: Encoder and decoder can have different `num_resblocks` and `base_channels`. The 32x config uses `encoder_num_resblocks=[2,3,9,2], decoder_num_resblocks=[3,4,12,2]` — the decoder is deeper per-stage (especially at Up2) to compensate for the lack of skip connections. Total params ≈ 478 M (enc 169 M / dec 309 M).

**HunyuanVAE-style I/O shortcut** (`use_io_shortcut=True` in config, default on): a second pair of non-parametric residual shortcuts bridging the narrowest point of the VAE.
- **Encoder output** (`Encoder._out_shortcut`): after mid block, `h.reshape(B, 2*latent_dim, C//(2*latent_dim), H, W).mean(dim=2)` — channel group-averaging, bypasses `norm_out → swish → conv_out` and adds to the post-conv output. For 32x config: `1024 → 128` (group_size=8).
- **Decoder input** (`Decoder._in_shortcut`): `z.repeat_interleave(base_channels[-1] // latent_dim, dim=1)` — channel repeat-interleave, bypasses `conv_in` and adds to its output before entering mid block. For 32x config: `64 → 1024` (repeats=16).
- Both are zero-parameter. Gated by `use_io_shortcut=True` in `WFIVAE2Model.__init__` (ConfigMixin-serialized). Requires `base_channels[-1] % (2*latent_dim) == 0` (encoder) and `base_channels[-1] % latent_dim == 0` (decoder); `__init__` asserts these when the flag is on.

### Channel Dimension Flow (32x config, 1024px input)

**Encoder** (spatial: 1024→512→256→128→64→32, channels: 3→256→512→512→1024→1024):
```
[3, 1024]  ->Wavelet->  [12, 512]  ->conv_in(12→256)->  [256, 512]
  ->WFDownBlock1 (num_rb=2): ResBlock(256) + DCAE shortcut + concat [+128 flow] + out_res_block  -> [512, 256]
  ->WFDownBlock2 (num_rb=3): ResBlock×2(512) + DCAE shortcut + concat [+128 flow] + out_res_block -> [512, 128]
  ->WFDownBlock3 (num_rb=9): ResBlock×8(512) + DCAE shortcut + concat [+128 flow] + out_res_block -> [1024, 64]
  ->WFDownBlock4 (num_rb=2): ResBlock(1024) + DCAE shortcut + concat [+128 flow] + out_res_block  -> [1024, 32]
  ->Mid(1024): ResBlock → Attention2DFix → ResBlock
  ->norm_out → SiLU → conv_out(1024→128)        -> [128, 32]     (128 = 2*latent_dim: μ and logvar)
  ->+ I/O shortcut: mid_out.reshape(B,128,8,32,32).mean(dim=2)    (non-parametric)
  ->latent [64, 32, 32]   (after posterior sampling)
```

**Decoder** (spatial: 32→64→128→256→512→1024, channels: 64→1024→1024→512→512→256→3):
```
[64, 32]  ->conv_in(64→1024)->  [1024, 32]  + I/O shortcut: z.repeat_interleave(16, dim=1)
  ->Mid(1024): ResBlock → Attention2DFix → ResBlock
  ->WFUpBlock1 (num_rb=3): branch(1024→1152) split [1024|128 flow] + ResBlock + up + DCAE shortcut + out_res -> [1024, 64]
  ->WFUpBlock2 (num_rb=4): branch(1024→1152) split [1024|128 flow] + ResBlock×2 + up + DCAE shortcut + out_res -> [512, 128]
  ->WFUpBlock3 (num_rb=12): branch(512→640) split [512|128 flow] + ResBlock×10 + up + DCAE shortcut + out_res -> [512, 256]
  ->WFUpBlock4 (num_rb=2): branch(512→640) split [512|128 flow] + up + DCAE shortcut + out_res            -> [256, 512]
  ->norm_out → SiLU → conv_out(256→12)          -> [12, 512]
  ->h[:, :3] += w_final (wavelet residual from last WFUpBlock)
  ->InverseHaarWaveletTransform2D              -> [3, 1024]
```

- **WFDownBlock energy flow**: wavelet coefficients (12ch) → `in_flow_conv` (12→`energy_flow_size`=128) → concatenated to main trunk before `out_res_block`.
- **WFUpBlock energy flow**: split 128 channels from branch output → `out_flow_conv` → 12ch wavelet coefficients → `InverseHaarWaveletTransform2D` → RGB residual `w` passed to the next WFUpBlock (or to the final conv_out merge).
- **Wavelet cascade**: encoder coefficients `enc_coeffs[0..3]` and decoder coefficients `dec_coeffs[0..3]` (reversed) feed the `wavelet_loss` L1 term at 4 scales.
- **Non-parametric shortcuts**: WFDownBlock / WFUpBlock each have an inner DCAE shortcut (space↔channel + group avg / dup), and the bottleneck has the I/O shortcut (group avg / repeat interleave). Zero params across all of them.

### Training Loop Design

`train_image_ddp.py` uses **alternating optimization** via `current_step % 2`: even steps (0, 2, 4...) optimize the generator, odd steps (1, 3, 5...) optimize the discriminator. The discriminator only activates when `current_step >= disc_start` (default 80000, this repo uses 5). Discriminator weight is computed adaptively based on gradient norms of the last decoder layer.

**Combined loss** (generator step):
```
generator_loss = rec_loss/exp(logvar) + logvar + perceptual_loss + kl_loss + disc_loss + wavelet_loss + fm_loss
```
where `disc_loss = -mean(D(recon)) * adaptive_weight * disc_factor`, `wavelet_loss = l1(encoder_coeffs, decoder_coeffs_reversed)`, and `fm_loss` is the pix2pixHD feature-matching loss (only active when `disc_type=multiscale`).

**Discriminator variants** (`--disc_type`, coexisting — no removal of either):
- `single` (default, legacy) — original single-scale PatchGAN (`NLayerDiscriminator`). `fm_loss=0`. Backward-compatible with every existing checkpoint.
- `multiscale` (pix2pixHD, CVPR 2018) — `num_D` independent PatchGANs on progressively `AvgPool2d`-downsampled inputs; `getIntermFeat=True` exposes per-layer activations for feature matching. `g_loss = mean over scales of -mean(D_i(recon))`; `fm_loss = lambda_feat * Σ_i Σ_j D_weights * feat_weights * L1(D_i(recon)[j], D_i(real)[j].detach())` where `feat_weights=4/(n_layers_D+1)`, `D_weights=1/num_D`. Extra cost: one additional `D(real)` forward pass per generator step. Multi-scale disc code is in `causalimagevae/model/losses/multiscale_discriminator.py`; the original single-scale disc in `discriminator.py` is **not touched**.

**Discriminator normalization** (`--disc_norm`, orthogonal to `--disc_type`, both discriminators support all four modes):
- `bn` (default, legacy) — `BatchNorm2d` on middle layers. Batch-statistics dependent; noisy at BS=2.
- `sn` (Miyato et al., ICLR 2018) — wraps every `Conv2d` with `torch.nn.utils.spectral_norm` (power iteration, Ip=1), replaces BN with `nn.Identity()`. Constrains Lipschitz constant per-layer; **batch-size independent**. state_dict gains `{layer}.weight_orig`, `{layer}.weight_u`, `{layer}.weight_v` keys (not cross-compatible with non-SN checkpoints). Helpers in `discriminator.py`: `_maybe_sn` / `_get_disc_norm_layer`. `weights_init` is patched to initialize `weight_orig` when SN is active.
- `in` — `InstanceNorm2d` (affine=True) on middle layers. Another batch-size-independent alternative, less common.
- `none` — `nn.Identity()` in place of BN, no SN. Pure Conv+LeakyReLU, for ablation only.

2 disc_type × 4 disc_norm = 8 combinations; `train_wfivae.sh` auto-suffixes `EXP_NAME` with `_{disc_type}_{disc_norm}` so each combination has its own output directory.

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
model = model_cls.from_config("examples/wfivae2-image-32x-192bc.json")
```

### Key Source Files

| Area | File |
|------|------|
| Model | `causalimagevae/model/vae/modeling_wfvae2.py` |
| Wavelet | `causalimagevae/model/modules/wavelet.py` |
| Loss/Discriminator | `causalimagevae/model/losses/perceptual_loss_2d.py` |
| Single-scale Discriminator | `causalimagevae/model/losses/discriminator.py` |
| Multi-scale Discriminator | `causalimagevae/model/losses/multiscale_discriminator.py` (pix2pixHD port) |
| EMA | `causalimagevae/model/ema_model.py` |
| Registry | `causalimagevae/model/registry.py` |
| DDP Sampler | `causalimagevae/dataset/ddp_sampler.py` |
| Dataset | `causalimagevae/dataset/image_dataset.py`, `manifest_dataset.py` |

## Key Training Parameters

| Parameter | Value |
|-----------|-------|
| `--model_name` | WFIVAE2 |
| `--model_config` | `examples/wfivae2-image-32x-192bc.json` — the only config in this repo (all legacy 8x configs were removed) |
| `--disc_cls` | causalimagevae.model.losses.LPIPSWithDiscriminator |
| `--disc_start` | 5 (this repo), 80000 (default) |
| `--disc_type` | `single` \| `multiscale` (default `single` in Python; `train_wfivae.sh` defaults to `multiscale`) |
| `--num_D` | 3 (multiscale only — number of discriminator scales) |
| `--n_layers_D` | 3 (multiscale only — conv layers per PatchGAN) |
| `--feat_match_weight` | 10.0 (multiscale only — pix2pixHD FM loss weight; 0 disables FM) |
| `--disc_norm` | `bn` \| `sn` \| `in` \| `none` (default `bn`). `sn` wraps every disc Conv2d with `torch.nn.utils.spectral_norm` and replaces BN with Identity (Miyato et al. ICLR 2018) — batch-size independent, best for BS=2 at 1024px. `in` uses InstanceNorm2d. `none` is pure Conv+LeakyReLU for ablation. |
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

CSV fields: `step, generator_loss, discriminator_loss, rec_loss, perceptual_loss, kl_loss, wavelet_loss, fm_loss, nll_loss, g_loss, d_weight, logits_real, logits_fake, nll_grads_norm, g_grads_norm, psnr, lpips, psnr_ema, lpips_ema, active_channels`. Output to `{ckpt_dir}/{exp_name}.csv`. `fm_loss` is always present (0 for single-scale disc). `logits_real`/`logits_fake` are averaged across scales when `disc_type=multiscale`.

Plot files: `training_curves.png` (at checkpoints), `training_curves_final.png` (normal completion), `training_curves_interrupted.png` (Ctrl+C). Features 7x3 subplot layout with smoothed curves and disc_start marker.

### Training Output Structure

Under `{ckpt_dir}/{exp_name-lr...-bs...-rs...}/`:
- `{exp_name}.csv` — metrics CSV
- `training_curves*.png` — loss plots
- `val_images/original/` and `val_images/reconstructed/` — validation images
- `checkpoint-*.ckpt` — model + optimizer state

## Model Configs

Only one config ships with this branch: **`examples/wfivae2-image-32x-192bc.json`**. All legacy 8x configs (`wfivae2-image-192bc.json`, `wfivae2-image-16chn.json`, `wfivae2-image-1024.json`, `wfivae2-image-64chn-192bc.json`) have been removed. Compression ratio is still `2^(len(base_channels))` for anyone porting a new config.

**Current config (`wfivae2-image-32x-192bc.json`):**
- `latent_dim=64`
- `base_channels=[256, 512, 512, 1024, 1024]` → **32x** spatial compression (2x wavelet + 16x from 4 downsampling blocks)
- `encoder_num_resblocks=[2, 3, 9, 2]` — per-stage depth, heavy middle at stage 2 (9 resblocks)
- `decoder_num_resblocks=[3, 4, 12, 2]` — per-stage depth, heavy middle at stage 2 (12 resblocks)
- `use_io_shortcut=true` — HunyuanVAE-style encoder-output and decoder-input shortcuts enabled
- Total params ≈ 478 M (encoder 169 M + decoder 309 M, non-parametric shortcuts add 0)

**Latent shape vs input resolution (32x):**
- 1024px input → `[B, 64, 32, 32]`
- 512px input → `[B, 64, 16, 16]`
- 256px input → `[B, 64, 8, 8]`

**`num_resblocks` semantics:** `num_resblocks=N` counts the full main-trunk ResBlocks per stage = 1 `branch_conv` (decoder only) + middle `res_block` seq + 1 `out_res_block`, so `WFDownBlock` middle depth = N−1 and `WFUpBlock` middle depth = N−2 (vs. DC-AE's `depth_list` which is the pure main-body count). `num_resblocks` accepts either a single int (legacy, uniform) or a list of length `len(base_channels)-1` indexed from the highest-res stage to the lowest-res. The current 32x setting of `[2,3,9,2]` / `[3,4,12,2]` gives middle-body depth of 8 / 10 at stage 2, matching DC-AE f32c32's Stage 2 depth.

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
| `RESOLUTION` | `256` | Image resolution — **only `256` or `1024` supported**; any other value makes `train_wfivae.sh` exit early with an error |
| `BATCH_SIZE` | `8` (256px) / `2` (1024px) | Per-GPU batch size |
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
| `EVAL_BATCH_SIZE` | `2` (1024px) / `8` (256px) | Validation batch size |
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
| `DISC_TYPE` | `multiscale` | Discriminator variant (`single` or `multiscale`). Launch script default flips to multiscale (pix2pixHD aggressive route); Python CLI default remains `single` for backward compat. `EXP_NAME` auto-suffixes with the type so single/multiscale runs land in separate output dirs. |
| `NUM_D` | `3` | Number of discriminator scales (multiscale only) |
| `N_LAYERS_D` | `3` | Conv layers per PatchGAN (multiscale only) |
| `FEAT_MATCH_WEIGHT` | `10.0` | pix2pixHD feature-matching loss weight (multiscale only; 0 disables FM while keeping multi-scale G/D loss) |
| `DISC_NORM` | `sn` | Discriminator normalization: `sn` (default, spectral_norm Miyato et al. 2018 — batch-size independent, recommended for BS=2 at 1024px), `bn` (legacy BatchNorm2d), `in` (InstanceNorm2d), `none` (Identity). `EXP_NAME` auto-suffixes with `_{disc_type}_{disc_norm}`, so every combo lands in its own output dir (e.g., `{project}_multiscale_sn/`). |
| `MIX_PRECISION` | `bf16` | Training precision: `bf16`, `fp16`, or `fp32` |

### Resume

| Variable | Default | Description |
|----------|---------|-------------|
| `RESUME_CKPT` | — | Path to checkpoint for resumption |

### Data Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `ORIGINAL_MANIFEST` | `/mnt/hpfs/HDU/ssk/SA-1B_256/train_manifest.jsonl` (256px) or `/mnt/hpfs/HDU/ssk/SA-1B/train_manifest.jsonl` (1024px, **246 server**) | Training JSONL manifest path |
| `VAL_MANIFEST` | Corresponding `val_manifest.jsonl` next to the training manifest | Pre-split validation manifest (skips auto-split) |
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
- **Discriminator checkpoints are not cross-compatible between `disc_type=single` and `disc_type=multiscale`** — state_dict shapes differ (single ≈ 2.77 M params, multiscale ≈ 8.30 M at `num_D=3, n_layers_D=3`). Resuming a checkpoint with the wrong `disc_type` raises a PyTorch shape mismatch. `train_wfivae.sh` auto-suffixes `EXP_NAME` with `_{disc_type}_{disc_norm}` so runs land in disjoint output dirs.
- **Discriminator checkpoints are not cross-compatible across `disc_norm` modes either** — `disc_norm=sn` adds `weight_orig`/`weight_u`/`weight_v` state_dict keys that `bn`/`in`/`none` don't have, and `bn`/`in` have affine norm params that `sn`/`none` lack. Switching mode requires training from scratch. The EXP_NAME suffix already isolates each combination.
- **`weights_init` initialization chain has an SN-aware branch** — when a module has `weight_orig` (PyTorch's old-API `spectral_norm` marker), init targets `weight_orig.data` instead of `weight.data` (which is a computed property after SN wrapping). Modifying `weights_init` must preserve both branches.

## Documentation Sync

When upgrading scripts or workflows, keep these docs in sync:
- `docs/README_image_training.md` (note: currently references outdated features like `val_patch_scores` and `training_losses.csv` — the CSV is now `{exp_name}.csv`)
- `docs/QUICK_REFERENCE.md` (note: references outdated paths and old config defaults)

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WF-VAE (Wavelet-driven energy Flow VAE) is an Image VAE architecture using multi-level Haar wavelet transforms to construct efficient energy flow pathways. Accepted by CVPR 2025. Spatial compression ratio is determined by `base_channels` length: `compression = 2^(len(base_channels))`. This branch ships **a single config** for a 32× image VAE (`base_channels=[256,512,512,1024,1024]`, `latent_dim=64`, `block_type="dcae"`) whose experiment **removes the mid-layer wavelet energy flow** (W^(2), W^(3) in the paper) — i.e., `use_energy_flow=false`. The encoder-input / decoder-output Haar WT (W^(1)) is always kept because it drives a 2× spatial compression. The `use_energy_flow` plumbing is preserved so flipping it back to `true` reproduces the paper form — handy for sanity-checking against the original design without maintaining a separate code path. A `block_type` switch and the classic stride-2/interp variants remain in the code (`WFDownBlockClassic` / `WFUpBlockClassic`) but no JSON ships for them. The model is fully convolutional — no positional encodings, no fixed spatial dimensions, so weights are resolution-agnostic.

## Common Commands

### Environment Setup
```bash
conda create -n wfvae python=3.10 -y
conda activate wfvae
pip install -r requirements.txt
```

### Training
```bash
# Only shipped config: DCAE block_type + mid-layer energy flow OFF
# (W^(1) HaarWT 保留, pixel_shuffle + non-parametric shortcuts + I/O shortcut + LayerNorm),
# multiscale+sn disc, ECA on
bash train_wfivae.sh

# Multi-GPU
GPU=0,1,2,3 bash train_wfivae.sh

# Resume
RESUME_CKPT="/path/to/checkpoint.ckpt" bash train_wfivae.sh

# Override resolution / batch size (RESOLUTION only supports 256 or 1024; default 256)
RESOLUTION=1024 BATCH_SIZE=1 EPOCHS=500 EVAL_STEPS=500 bash train_wfivae.sh

# Gradient accumulation (default: 4 @ 1024 / 1 @ 256; effective batch = BATCH_SIZE * NGPU * GRAD_ACCUM_STEPS)
GRAD_ACCUM_STEPS=8 BATCH_SIZE=1 bash train_wfivae.sh

# Discriminator ablations (default is multiscale + sn)
DISC_TYPE=single DISC_NORM=bn bash train_wfivae.sh       # legacy single-scale PatchGAN with BN
DISC_TYPE=multiscale FEAT_MATCH_WEIGHT=0 bash train_wfivae.sh  # multiscale but disable FM loss

# Direct torchrun (low-level, requires manual dataset setup)
torchrun --nproc_per_node=8 train_image_ddp.py \
    --exp_name WFIVAE_1024 \
    --image_path /path/to/train_manifest.jsonl --use_manifest \
    --eval_image_path /path/to/val_manifest.jsonl \
    --model_name WFIVAE2 \
    --model_config examples/wfivae2-image-32x-192bc.json \
    --resolution 1024 --batch_size 1 --lr 1e-5 \
    --ema --wavelet_loss --eval_lpips
```

### Testing
```bash
# Forward/backward sanity check (reads block_type/use_eca/use_io_shortcut/use_energy_flow from JSON)
python scripts/test_imagevae.py --config examples/wfivae2-image-32x-192bc.json
```

### Pre-commit Validation
```bash
python3 -m py_compile train_image_ddp.py causalimagevae/model/vae/modeling_wfvae2.py
bash -n train_wfivae.sh
```

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
- `scripts/recon_single_image_flux.py` — FLUX.2 VAE reconstruction (comparison baseline)
- `scripts/draw_architecture.py` — Matplotlib architecture diagram generator (outputs `wfvae2_32x_dcae_architecture.png` + `wfvae2_32x_dcae_blocks_detail.png`). **Note: only draws the DCAE variant.**
- `architecture_diagram.html` — Interactive mermaid-based architecture reference. Open in browser.
- `scripts/resize_sa1b_256.py` / `resize_sa1b_512.py` / `resize_faces_to_256_512.py` — Dataset preprocessing tools

## Architecture

### Wavelet Energy Flow Pipeline

Core innovation: explicit frequency-domain information flow via cascaded Haar wavelets.

- **Encoder**: Input → `HaarWaveletTransform2D` (RGB → 12 coeffs: 4 sub-bands × 3 ch, at half resolution) → `WFDownBlock` × N (merge wavelet energy flow with spatial features via concat, then downsample) → mid block → latent distribution
- **Decoder**: Latent → mid block → `WFUpBlock` × N (split energy flow, generate wavelet coefficients via `out_flow_conv`, upsample) → `InverseHaarWaveletTransform2D` → output

Key: the decoder is fully independent — no skip connections from encoder. Energy flow is reconstructed entirely from the latent.

### `block_type`: DCAE vs Classic spatial transitions (key switch)

The JSON config field `block_type` (`"dcae"` | `"classic"`) selects one of two implementations of `WFDownBlock` / `WFUpBlock`. The **shell default is `"dcae"`** (via `DEFAULT_MODEL_CONFIG="examples/wfivae2-image-32x-192bc.json"` in `train_wfivae.sh`), matching the Python class-level default. These are **completely separate classes** in `causalimagevae/model/vae/modeling_wfvae2.py`:

- **`WFDownBlock` + `WFUpBlock` (DCAE, shell default)** — Spatial transition uses `Conv(in, out/4, s=1) + F.pixel_unshuffle(2)` (down) / `Conv(in, out*4, s=1) + F.pixel_shuffle(2)` (up). Each is wrapped with a non-parametric **DCAE residual shortcut**: `_down_shortcut = F.pixel_unshuffle(x,2) → reshape-group-mean` and `_up_shortcut = repeat_interleave → F.pixel_shuffle(x,2)`. Zero-param shortcuts bypass the main conv trunk to improve gradient flow. Also the Python class default. Config: `examples/wfivae2-image-32x-192bc.json`.
- **`WFDownBlockClassic` + `WFUpBlockClassic` (A/B alternate)** — Spatial transition uses the modules from `causalimagevae/model/modules/updownsample.py`: `Downsample` = `F.pad(...,(0,1,0,1)) + Conv(s=2, p=0)`, `Upsample` = `F.interpolate(nearest, 2×) + Conv(s=1)`. **No shortcut**. Both classic modules keep channels constant and let the `out_res_block` handle the channel count change. This is the LDM/VQGAN-era spatial transition, kept as an option for comparison. Switch to it via `MODEL_CONFIG=examples/wfivae2-image-32x-192bc-classic.json bash train_wfivae.sh`. The migrating colored high-frequency artifacts previously observed on DCAE at 32× were traced to **GroupNorm** (not `pixel_shuffle` itself); switching the DCAE config to `norm_type="layernorm"` eliminates them, so DCAE + LayerNorm now trains cleanly. The classic config retains `norm_type="groupnorm"` since the `interpolate+conv` transition tolerates GroupNorm without the same failure mode.

`Encoder.__init__` and `Decoder.__init__` each receive `block_type` and pick `down_block_cls = WFDownBlock if block_type == "dcae" else WFDownBlockClassic` (symmetrically for up). **The two implementations have different state_dict layouts** (`conv_down.weight` / `conv_up.weight` vs `down.conv.weight` / `up.conv.weight`) — checkpoints are not cross-compatible.

### I/O shortcut coupling with `block_type`

`use_io_shortcut` in the JSON enables a second pair of non-parametric bottleneck shortcuts (HunyuanVAE-style):

- **Encoder output** (`Encoder._out_shortcut`): after mid block, `h.reshape(B, 2*latent_dim, C//(2*latent_dim), H, W).mean(dim=2)` — channel group-averaging, bypasses `norm_out → swish → conv_out`. For 32× config: 1024 → 128 (group_size=8).
- **Decoder input** (`Decoder._in_shortcut`): `z.repeat_interleave(base_channels[-1]//latent_dim, dim=1)` — channel repeat-interleave, bypasses `conv_in`. For 32× config: 64 → 1024 (repeats=16).

Both are zero-parameter. **The I/O shortcut is a DCAE-era concept and only makes sense with `block_type="dcae"`.** If a user sets `block_type="classic"` AND `use_io_shortcut=True`, `WFIVAE2Model.__init__` **soft-downgrades** `use_io_shortcut=False`, prints a warning, and writes the downgrade back via `self.register_to_config(use_io_shortcut=False)` so the saved `config.json` reflects reality. Requires `base_channels[-1] % (2*latent_dim) == 0` (encoder) and `base_channels[-1] % latent_dim == 0` (decoder) when enabled; init-time asserts enforce this.

### ECA channel attention

`use_eca` gates `ECALayer` (Wang et al., CVPR 2020) at the end of every `ResnetBlock2D`, applied to post-`conv2` activation before the residual add. ECA is parameter-light vs SE — no channel reduction, just a 1-D conv over the GAP descriptor with adaptive kernel `k = |log2(C)/2 + 0.5|_odd`. For this 32× config (C ∈ {256, 512, 1024}) k=5 across the board, so each ECA layer adds **only 5 trainable params**. Module: `causalimagevae/model/modules/eca.py`. The flag is plumbed through Encoder/Decoder/WFDown(Classic)/WFUp(Classic)/ResnetBlock2D. **Works with both `block_type` variants**. Flipping `use_eca` changes the state_dict (adds/removes `*.eca.conv.weight` keys), so checkpoints are not cross-compatible. `train_wfivae.sh` auto-suffixes `EXP_NAME` with `_eca` when the JSON has `use_eca=true`.

### `use_energy_flow`: mid-layer wavelet pathway on/off

`use_energy_flow` in the JSON toggles the **mid-layer** wavelet inflow/outflow pathway (論文的 W^(2)、W^(3)… energy flow pathway). The encoder-input / decoder-output Haar WT (論文 W^(1)) is **always kept** — it drives the 2× spatial compression at the encoder entry and is part of the backbone, not the bypass.

- **`use_energy_flow=true` (default, paper form)** — Each `WFDownBlock` / `WFDownBlockClassic` runs `wavelet_transform(w[:,:3]) → in_flow_conv(12→128) → concat into the trunk`; `out_res_block`'s `in_channels = out_ch + 128` (dcae) or `in_ch + 128` (classic). Each `WFUpBlock` / `WFUpBlockClassic` has `branch_conv` widen channels to `in_ch + 128`, splits off the `-128:` slice through `out_flow_conv → IWT` to produce a wavelet residual `w` that cascades into the next stage; main trunk only uses the first `in_ch` channels. Encoder returns `inter_coeffs` (list of 4 tensors), decoder consumes/returns them, `WFIVAE2Model.forward.extra_output = (enc_coeffs, dec_coeffs)` — train loop feeds these into `wavelet_loss`.
- **`use_energy_flow=false`** — All of the above is dropped: no `wavelet_transform` / `in_flow_conv` / `out_flow_conv` / `inverse_wavelet_transform` **inside** the blocks; `branch_conv` outputs `in_ch` (no `+128` expansion); `out_res_block` input drops back to the pre-expansion width. `Encoder.forward` returns `(h, None)`, `Decoder.forward` returns `(dec, None)`, and `WFIVAE2Model.forward` collapses to **`extra_output = None`** (not `(None, None)` — that's truthy in Python and would mis-trigger the WL loss branch). The `W^(1)` Haar WT at the encoder entry and IWT at the decoder exit are still applied unchanged, so compression ratio stays at 32×.

**Zero changes** to `train_image_ddp.py` and `perceptual_loss_2d.py`:
- `train_image_ddp.py:1437-1439` guards `wavelet_coeffs = outputs.extra_output` on `outputs.extra_output is not None` — automatically `None` when ef is off.
- `perceptual_loss_2d.py:206-217` guards WL loss on `if wavelet_coeffs:` — falls back to `wl_loss = torch.tensor(0.0)` when `None`, so CSV `wavelet_loss` column shows 0 and no gradient flows through a non-existent pathway.

Flipping `use_energy_flow` **changes the generator state_dict** (adds/removes `*.in_flow_conv.*`, `*.out_flow_conv.*`, and the `+128` channel expansion in `branch_conv` weights), so checkpoints are not cross-compatible. `train_wfivae.sh` auto-suffixes `EXP_NAME` with `_noef` when the JSON has `use_energy_flow=false`, so each ef-on/off run lands in its own output directory.

**The `_noef` flag is orthogonal to `block_type`, `use_eca`, and `use_io_shortcut`**, giving a 2×2×2×2 matrix of configurable variants (only a few are shipped as presets).

### Channel Dimension Flow (32× config, 1024px input, DCAE variant)

**Encoder** (spatial: 1024→512→256→128→64→32, channels: 3→256→512→512→1024→1024):
```
[3, 1024]  ->Wavelet->  [12, 512]  ->conv_in(12→256)->  [256, 512]
  ->WFDownBlock1 (num_rb=4): ResBlock×3(256) + DCAE shortcut + concat [+128 flow] + out_res_block  -> [512, 256]
  ->WFDownBlock2 (num_rb=4): ResBlock×3(512) + DCAE shortcut + concat [+128 flow] + out_res_block  -> [512, 128]
  ->WFDownBlock3 (num_rb=4): ResBlock×3(512) + DCAE shortcut + concat [+128 flow] + out_res_block  -> [1024, 64]
  ->WFDownBlock4 (num_rb=2): ResBlock×1(1024) + DCAE shortcut + concat [+128 flow] + out_res_block -> [1024, 32]
  ->Mid(1024): ResBlock → Attention2DFix → ResBlock
  ->(optional) I/O shortcut branch: mid_out.reshape(B,128,8,32,32).mean(dim=2)  (non-parametric)
  ->norm_out → SiLU → conv_out(1024→128)        -> [128, 32]   (+= io_shortcut if on)
  ->latent [64, 32, 32]   (after posterior sampling, 128=2*latent_dim split into μ and logvar)
```

**Decoder** (spatial: 32→64→128→256→512→1024, channels: 64→1024→1024→512→512→256→3):
```
[64, 32]  ->conv_in(64→1024)  (+ optional I/O shortcut: z.repeat_interleave(16, dim=1))  -> [1024, 32]
  ->Mid(1024): ResBlock → Attention2DFix → ResBlock
  ->WFUpBlock1 (num_rb=5): branch(1024→1152) split [1024|128 flow] + ResBlock×3 + up + DCAE shortcut + out_res -> [1024, 64]
  ->WFUpBlock2 (num_rb=5): branch(1024→1152) split [1024|128 flow] + ResBlock×3 + up + DCAE shortcut + out_res -> [512, 128]
  ->WFUpBlock3 (num_rb=5): branch(512→640) split [512|128 flow] + ResBlock×3 + up + DCAE shortcut + out_res   -> [512, 256]
  ->WFUpBlock4 (num_rb=3): branch(512→640) split [512|128 flow] + ResBlock×1 + up + DCAE shortcut + out_res   -> [256, 512]
  ->norm_out → SiLU → conv_out(256→12)          -> [12, 512]
  ->h[:, :3] += w_final (wavelet residual from last WFUpBlock)
  ->InverseHaarWaveletTransform2D              -> [3, 1024]
```

- **WFDownBlock energy flow**: wavelet coefficients (12ch) → `in_flow_conv` (12→`energy_flow_size=128`) → concatenated to main trunk before `out_res_block`.
- **WFUpBlock energy flow**: split 128 channels from branch output → `out_flow_conv` → 12 wavelet coefs → `InverseHaarWaveletTransform2D` → RGB residual `w` passed to the next WFUpBlock (or the final conv_out merge).
- **Wavelet cascade**: encoder coeffs `enc_coeffs[0..3]` and decoder coeffs `dec_coeffs[0..3]` (reversed) feed the `wavelet_loss` L1 term at 4 scales.
- **Asymmetric encoder/decoder**: encoder and decoder can have different `num_resblocks` and `base_channels`. The 32× config uses `encoder_num_resblocks=[4,4,4,2], decoder_num_resblocks=[5,5,5,3]` — shallow at the deepest stage (Down3/Up3) to cap activation memory at 1024px.
- **Classic variant** (`block_type="classic"`) follows the same data-flow topology but substitutes `Downsample`/`Upsample` for the DCAE `conv_down/pixel_unshuffle` + shortcut pair, and ends with different `out_res_block` input channels (`in_ch + ef` instead of `out_ch + ef` for down; `in_ch → out_ch` handled by `out_res_block` for up).

### Parameter counts (current 32× config)

| Variant | Encoder | Decoder | Total |
|---|---|---|---|
| DCAE + no-ef (shipped) | 145.55 M | 322.25 M | **467.80 M** |
| DCAE + ef (paper form — flip `use_energy_flow=true` in the JSON to reproduce) | 152.17 M | 337.71 M | 489.88 M |

Relative to the paper-form DCAE + ef (489.88 M), the shipped DCAE + no-ef saves ~22 M (−4.5%) — savings come from dropping `in_flow_conv` (12→128) in each down-block, the full `out_flow_conv` (128→128 ResBlock + 128→12 conv) in each up-block, and the `+128` channel expansion in decoder `branch_conv`s.

### Training Loop Design

`train_image_ddp.py` uses **alternating optimization** via `current_step % 2`: even steps (0, 2, 4...) optimize the generator, odd steps optimize the discriminator — **from step 0, independent of `disc_start`**. `disc_start` (shell & Python default: 80000) only gates the **generator** side of the GAN: while `current_step < disc_start`, the generator's `g_loss` and `fm_loss` are both forced to zero and `d_weight` is 0, so only L1 + LPIPS + KL (+ wavelet if enabled) drive the generator; in parallel, the discriminator is still trained on real / L1-only-reconstruction pairs so it's calibrated when adversarial gradient starts flowing at step `disc_start`. After the switch, discriminator contribution is weighted by `adaptive_weight = ||nll_grads|| / ||g_grads||`, clamped to `adaptive_weight_clamp` (default 1e6), then multiplied by `disc_weight` (0.5).

**Combined generator loss**:
```
total = weighted_nll_loss + kl_weight*kl_loss + d_weight*disc_factor*g_loss + wavelet_weight*wl_loss + fm_loss
```
where `disc_loss = -mean(D(recon))`, `wl_loss = l1(encoder_coeffs, decoder_coeffs_reversed)`, `fm_loss` is the pix2pixHD feature-matching loss (only active when `disc_type="multiscale"`). **Note**: `nll_loss` in `perceptual_loss_2d.py:190-197` is the per-pixel loss times an unreduced sum over H×W (`torch.sum(nll_loss)/batch`), so at 1024² its magnitude is ~3M× the per-pixel mean, which causes `adaptive_weight` to hit the clamp. This is by design and applies equally to any subclass of `LPIPSWithDiscriminator`.

### Discriminator variants (`--disc_type`)

- `single` (legacy) — original single-scale PatchGAN (`NLayerDiscriminator`). `fm_loss=0` regardless of `--feat_match_weight`.
- `multiscale` (**default**, pix2pixHD CVPR 2018) — `num_D` independent PatchGANs on progressively `AvgPool2d`-downsampled inputs; exposes per-layer activations for feature matching. `g_loss = mean over scales of -mean(D_i(recon))`; `fm_loss = lambda_feat * Σ_i Σ_j D_weights * feat_weights * L1(D_i(recon)[j], D_i(real)[j].detach())`. Extra cost: one additional `D(real)` forward pass per generator step. Code: `causalimagevae/model/losses/multiscale_discriminator.py`. The single-scale disc in `discriminator.py` is **untouched** and still works via `DISC_TYPE=single`.

### Discriminator normalization (`--disc_norm`, orthogonal to `--disc_type`)

- `sn` (**default**, Miyato et al. ICLR 2018) — wraps every `Conv2d` with `torch.nn.utils.spectral_norm` (power iteration, Ip=1), replaces BN with `nn.Identity()`. Constrains per-layer Lipschitz constant; **batch-size independent**, best at BS=1/2 for 1024px. state_dict gains `{layer}.weight_orig`, `{layer}.weight_u`, `{layer}.weight_v`.
- `bn` (legacy) — `BatchNorm2d` on middle layers. Noisy at BS=2 without SyncBN.
- `in` — `InstanceNorm2d` (affine=True). Another batch-size-independent alternative.
- `none` — `nn.Identity()` in place of BN, no SN. Pure Conv+LeakyReLU, ablation only.

`weights_init` in `discriminator.py` has an SN-aware branch: when a module has `weight_orig`, init targets `weight_orig.data` instead of `weight.data` (which is a computed property after SN wrapping).

### Other training knobs

- **Gradient accumulation** (`--gradient_accumulation_steps N`): uses `model.no_sync()` / `disc.no_sync()` on non-final micro-batches to skip DDP all-reduce. Loss is divided by N. The step type (gen/disc) is determined at the start of each accumulation cycle and held constant across micro-batches. Partial accumulation at epoch boundaries is discarded.
- **TTUR** (`--disc_lr`): discriminator can use a separate learning rate (default = `--lr`).
- **Learned logvar** (`--learn_logvar`): makes `logvar` a trainable scalar with its own param group (lr controlled by `--logvar_lr`, default 1e-2), auto-balancing recon loss vs GAN loss magnitude. Off by default.
- **Mixed precision**: `--mix_precision bf16` (default), also supports `fp16` (uses `GradScaler`) and `fp32`.
- **Signal handling**: Ctrl+C triggers graceful shutdown — saves `training_curves_interrupted.png` and exits cleanly on rank 0.
- **Image normalization**: input/output in [-1, 1] range (torchvision `Normalize([0.5]*3, [0.5]*3)`).

### Checkpoint Format

Keys: `gen_model`, `dics_model` (**historical typo** for discriminator — do NOT rename, it would break checkpoint resumption), `optimizer_state`, `ema_state_dict`, `scaler_state`, `sampler_state`. When loading for inference, EMA weights are preferred over the normal state_dict. `WFIVAE2Model.init_from_ckpt` calls the `_warn_block_type_mismatch` static method to detect when a ckpt's key names (e.g. `conv_down.weight` vs `down.conv.weight`) don't match `self.config["block_type"]`, printing a diagnostic warning.

### Registry Pattern

```python
model_cls = ModelRegistry.get_model("WFIVAE2")
model = model_cls.from_pretrained("/path/to/checkpoint")
# or from config
model = model_cls.from_config("examples/wfivae2-image-32x-192bc.json")
```

### Key Source Files

| Area | File |
|------|------|
| Model | `causalimagevae/model/vae/modeling_wfvae2.py` (contains `WFDownBlock`, `WFUpBlock`, `WFDownBlockClassic`, `WFUpBlockClassic`, `Encoder`, `Decoder`, `WFIVAE2Model`) |
| Wavelet | `causalimagevae/model/modules/wavelet.py` |
| Classic up/down | `causalimagevae/model/modules/updownsample.py` (used only by the `*Classic` blocks) |
| Loss | `causalimagevae/model/losses/perceptual_loss_2d.py` (`LPIPSWithDiscriminator`, handles both `disc_type` branches) |
| Single-scale Disc | `causalimagevae/model/losses/discriminator.py` (untouched legacy) |
| Multi-scale Disc | `causalimagevae/model/losses/multiscale_discriminator.py` (pix2pixHD port) |
| ECA | `causalimagevae/model/modules/eca.py` |
| EMA | `causalimagevae/model/ema_model.py` |
| Registry | `causalimagevae/model/registry.py` |
| DDP Sampler | `causalimagevae/dataset/ddp_sampler.py` |
| Dataset | `causalimagevae/dataset/image_dataset.py`, `manifest_dataset.py` |

## Key Training Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--model_name` | `WFIVAE2` | |
| `--model_config` | `examples/wfivae2-image-32x-192bc.json` | only shipped config (DCAE + no-ef, this branch's experiment) |
| `--disc_cls` | `causalimagevae.model.losses.LPIPSWithDiscriminator` | |
| `--disc_start` | 80000 (both shell and Python) | Step at which the **generator** starts consuming GAN/FM loss. The **discriminator itself trains from step 0** regardless of this flag (see Training Loop Design). |
| `--disc_type` | `multiscale` | `single` \| `multiscale`. FM loss is gated on `multiscale`; `single` runs always have `fm_loss=0`. |
| `--disc_norm` | `sn` | `bn` \| `sn` \| `in` \| `none`. Both Python and shell default to `sn`. |
| `--num_D` | 3 | multiscale only |
| `--n_layers_D` | 3 | multiscale only |
| `--feat_match_weight` | 10.0 | multiscale only; 0 disables FM while keeping multi-scale hinge |
| `--kl_weight` | 1e-6 | |
| `--wavelet_weight` | 0.1 | |
| `--disc_weight` | 0.5 | |
| `--perceptual_weight` | 1.0 | |
| `--loss_type` | `l1` | |
| `--adaptive_weight_clamp` | 1e6 | Upper bound on the adaptive `d_weight` |

`train_wfivae.sh` auto-suffixes `EXP_NAME` with `_{disc_type}_{disc_norm}`, optionally `_eca` (if `use_eca` in JSON), optionally `_classic` (if `block_type=="classic"` in JSON), and optionally `_noef` (if `use_energy_flow==false` in JSON). The `_classic` branch of the suffix logic is dead weight for this branch (no classic JSON ships) but kept because the model code still supports `block_type="classic"` — if you add a custom classic JSON later, the suffix automation just works. **Default shell EXP_NAME (shipped DCAE + no-ef config)**: `{project}_multiscale_sn_eca_noef`. If you flip the JSON's `use_energy_flow` to `true` to reproduce the paper form on this backbone, the `_noef` suffix drops and you get `{project}_multiscale_sn_eca`.

### Loss Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv_log_steps` | 50 | Log losses to CSV every N steps |
| `--disable_plot` | False | Disable automatic plot generation |
| `--disable_wandb` | False | Disable WandB logging (`wandb` is optional import) |
| `--eval_subset_size` | 30 | Validation subset size (0 = full set) |
| `--eval_num_image_log` | 20 | Number of validation images to save |

CSV fields: `step, generator_loss, discriminator_loss, rec_loss, perceptual_loss, kl_loss, wavelet_loss, fm_loss, nll_loss, g_loss, d_weight, logits_real, logits_fake, nll_grads_norm, g_grads_norm, psnr, lpips, psnr_ema, lpips_ema, active_channels`. Output: `{ckpt_dir}/{exp_name}.csv`. `fm_loss` is always present (0 for single-scale disc). `logits_real`/`logits_fake` are averaged across scales when `disc_type=multiscale`.

Plot files: `training_curves.png` (at checkpoints), `training_curves_final.png` (normal completion), `training_curves_interrupted.png` (Ctrl+C). 7×3 subplot layout with smoothed curves and a `disc_start` marker.

### Training Output Structure

Under `{ckpt_dir}/{exp_name-lr...-bs...-rs...}/`:
- `README.md` — human-readable snapshot of this run's args, model JSON config, launch command, distributed env, software/git info. Written on rank 0 right after wandb init. On resume, a new `## 第 N 次运行 @ timestamp` section is **appended** (not overwritten); HTML-comment markers `<!-- run N @ ... -->` serve as invisible counters.
- `{exp_name}.csv` — metrics CSV
- `training_curves*.png` — loss plots
- `val_images/original/` and `val_images/reconstructed/` — validation images
- `checkpoint-*.ckpt` — model + optimizer state

## Model Configs

Exactly one config ships: `examples/wfivae2-image-32x-192bc.json` (DCAE + no-ef) — this branch's experiment. 32× compression via W^(1) Haar WT (2×) + 4 DCAE down stages (16×).

- `latent_dim=64`
- `base_channels=[256, 512, 512, 1024, 1024]`
- `encoder_num_resblocks=[4, 4, 4, 2]` / `decoder_num_resblocks=[5, 5, 5, 3]`
- `use_io_shortcut=true`, `use_eca=true`, `block_type="dcae"`, `norm_type="layernorm"`, **`use_energy_flow=false`**
- Uses DCAE `Conv + pixel_unshuffle / pixel_shuffle + non-parametric shortcut` for spatial transitions. Mid-layer wavelet inflow/outflow pathway is disabled; W^(1) Haar WT at encoder entry and IWT at decoder exit are retained (so 32× compression is preserved).
- **LayerNorm is load-bearing**: switching to `groupnorm` reintroduces migrating colored high-frequency artifacts at 32× under GAN pressure (originally traced to **GroupNorm interacting badly with `pixel_shuffle`**).
- **Total: 467.80 M params** (encoder 145.55 M + decoder 322.25 M).
- Paper-form sanity A/B: flip this JSON to `"use_energy_flow": true` — the model code still supports it; EXP_NAME drops the `_noef` suffix accordingly. No separate config ships.

**Latent shape vs input resolution (both variants, 32×):**
- 1024px input → `[B, 64, 32, 32]`
- 512px input → `[B, 64, 16, 16]`
- 256px input → `[B, 64, 8, 8]`

**`num_resblocks` semantics:** `num_resblocks=N` counts the full main-trunk ResBlocks per stage = 1 `branch_conv` (decoder only) + middle `res_block` seq + 1 `out_res_block`, so `WFDownBlock` middle depth = N−1 and `WFUpBlock` middle depth = N−2. Accepts either a single int (legacy, uniform) or a list of length `len(base_channels)-1` indexed from the highest-res stage to the lowest-res. Classic variants share the same `num_res_blocks` semantics.

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
| `RESOLUTION` | `256` | **Only `256` or `1024` supported**; any other value exits with error |
| `BATCH_SIZE` | `1` (1024px) / `8` (256px) | Per-GPU batch size, defaulted by resolution |
| `EPOCHS` | `1000` | Training epochs |
| `EVAL_STEPS` | `1000` | Validation interval |
| `SAVE_CKPT_STEP` | `10000` | Checkpoint save interval |
| `EVAL_SUBSET_SIZE` | `30` | Validation subset (0 = full) |
| `EVAL_NUM_IMAGE_LOG` | `20` | Val images to save & visualize |
| `CSV_LOG_STEPS` | `50` | CSV logging frequency |
| `LOG_STEPS` | `10` | Console logging frequency |
| `DATASET_NUM_WORKER` | `8` | DataLoader workers |
| `EXP_NAME` | auto (see below) | Output dir + CSV filename |
| `MODEL_CONFIG` | `examples/wfivae2-image-32x-192bc.json` | Only shipped config (DCAE + no-ef) |
| `MAX_STEPS` | — | Stop after N steps (overrides `EPOCHS`) |
| `EVAL_BATCH_SIZE` | `1` (1024px) / `8` (256px) | Validation batch size |
| `WANDB_PROJECT` | `WFIVAE` | WandB project name |
| `MASTER_PORT` | auto-detect 29500-29599 | torchrun master port |

### Learning Rate & Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `LR` | `1e-5` | Generator learning rate |
| `DISC_LR` | same as `LR` | Discriminator LR (TTUR: set 2–4× higher than LR) |
| `GRAD_ACCUM_STEPS` | `4` (1024px) / `1` (256px) | Effective batch = BATCH_SIZE × NGPU × this |
| `LEARN_LOGVAR` | — | Non-empty to enable learned logvar |
| `LOGVAR_LR` | `1e-2` | Learning rate for logvar parameter |
| `ADAPTIVE_WEIGHT_CLAMP` | `1e6` | Max adaptive weight for discriminator |
| `DISC_TYPE` | `multiscale` | `single` or `multiscale`. Auto-suffixes EXP_NAME. |
| `NUM_D` | `3` | Discriminator scales (multiscale only) |
| `N_LAYERS_D` | `3` | Conv layers per PatchGAN (multiscale only) |
| `FEAT_MATCH_WEIGHT` | `10.0` | pix2pixHD FM loss weight (multiscale only; 0 disables FM while keeping multi-scale G/D) |
| `DISC_NORM` | `sn` | `sn` \| `bn` \| `in` \| `none`. Auto-suffixes EXP_NAME. |
| `MIX_PRECISION` | `bf16` | `bf16`, `fp16`, or `fp32` |

### Resume

| Variable | Default | Description |
|----------|---------|-------------|
| `RESUME_CKPT` | — | Path to checkpoint for resumption |

### Data Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `ORIGINAL_MANIFEST` | `/mnt/hpfs/HDU/ssk/SA-1B_256/train_manifest.jsonl` (256px) or `/mnt/hpfs/HDU/ssk/SA-1B/train_manifest.jsonl` (1024px, **246 server**) | Training JSONL manifest |
| `VAL_MANIFEST` | corresponding `val_manifest.jsonl` | Pre-split validation manifest |
| `OUTPUT_DIR` | `/mnt/sdb/ssk/Exp_output/{project_name}` | Output directory |
| `DISABLE_WANDB` | `1` | `1`/`true`/`yes` disables WandB |
| `TRAIN_RATIO` | `0.9` | Train/val split ratio (only when no VAL_MANIFEST) |

## Distributed Training Notes

- PyTorch DDP with NCCL backend; `train_image_ddp.py` always requires DDP environment (use `torchrun` even for single-GPU)
- `--find_unused_parameters` is on by default (needed because gen/disc alternation means some params have no grad on each step)
- Set `CUDA_VISIBLE_DEVICES` before `torchrun`
- Checkpoints and WandB logging on rank 0 only
- Validation results gathered across all ranks via `dist.all_reduce` and `all_gather_object`
- Gradient accumulation uses `model.no_sync()` to skip all-reduce on non-final micro-batches

## High-Risk Modification Points

- DDP gather logic in `train_image_ddp.py` uses `all_gather_object` — changing validation return structure requires updating the gather/logging/save chain together.
- Checkpoint discriminator key is `dics_model` (typo) — renaming breaks old checkpoint loading.
- Do not modify data paths to your local private paths without confirmation.
- Gradient accumulation interacts with gen/disc alternation: step type is locked for the entire accumulation cycle, so each logical step still alternates correctly.
- `ValidImageDataset` and `ValidManifestImageDataset` both provide an `index` field for deduplication during validation gather — do not drop it.
- When modifying generator loss path, `learn_logvar`, or DDP wrapper boundaries, also check `aux_gen_params` sync logic (manual gradient sync for non-DDP-wrapped params).
- `sampler_state` in checkpoint stores "consumed batch position" — changing ckpt save/resume must verify epoch boundary recovery doesn't replay samples.
- **`block_type` DCAE vs classic checkpoints are not cross-compatible** — state_dict key names differ (`.conv_down.weight` / `.conv_up.weight` vs `.down.conv.weight` / `.up.conv.weight`). `_warn_block_type_mismatch` in `init_from_ckpt` detects this and warns, but `load_state_dict(strict=False)` in `train_image_ddp.py` will silently ignore mismatched keys at resume time — the `_classic` EXP_NAME suffix is the first line of defense against this footgun.
- **`use_io_shortcut` is soft-downgraded when `block_type="classic"`** — `WFIVAE2Model.__init__` forces it to False, prints a warning, and calls `self.register_to_config(use_io_shortcut=False)`. Don't remove either step; the `register_to_config` call ensures the saved `config.json` reflects the runtime state, so re-loading the ckpt doesn't reintroduce the conflict.
- **Discriminator checkpoints are not cross-compatible between `disc_type=single` and `disc_type=multiscale`** — state_dict shapes differ (~2.77 M vs ~8.30 M params at `num_D=3, n_layers_D=3`). Resuming a checkpoint with the wrong `disc_type` raises a PyTorch shape mismatch. The `_{disc_type}_{disc_norm}` EXP_NAME suffix isolates runs.
- **Discriminator checkpoints are not cross-compatible across `disc_norm` modes either** — `sn` adds `weight_orig`/`weight_u`/`weight_v` keys that `bn`/`in`/`none` don't have; `bn`/`in` have affine norm params that `sn`/`none` lack. Switching requires training from scratch.
- **`weights_init` has an SN-aware branch** — when a module has `weight_orig` (PyTorch's old-API `spectral_norm` marker), init targets `weight_orig.data` instead of `weight.data` (which is a computed property after SN wrapping). Preserve both branches.
- **ECA on/off changes the generator state_dict structure** — flipping `use_eca` adds/removes `*.eca.conv.weight` keys (5 params per ResBlock). Resuming with the wrong `use_eca` raises a state_dict key mismatch on `gen_model` load. The `_eca` EXP_NAME suffix isolates runs.
- **`use_energy_flow` on/off changes the generator state_dict structure** — flipping the flag adds/removes `*.in_flow_conv.*`, `*.out_flow_conv.*`, `*.inverse_wavelet_transform.*`, and shifts `branch_conv.*` / `out_res_block.norm1.*` / `out_res_block.conv1.*` channel dimensions (the `+energy_flow_size` expansion collapses). Resuming with the wrong `use_energy_flow` raises a state_dict shape/key mismatch; `load_state_dict(strict=False)` silently drops mismatches, so the `_noef` EXP_NAME suffix is the first line of defense. W^(1)-side modules (`encoder.wavelet_transform_in`, `decoder.inverse_wavelet_transform_out`, `encoder.conv_in` 12-ch input, `decoder.conv_out` 12-ch output) are preserved across both flag states.
- **`WFIVAE2Model.forward.extra_output` must be `None` (not `(None, None)`) when `use_energy_flow=False`** — a `(None, None)` tuple is truthy, which would mis-trigger both `if outputs.extra_output is not None` (train loop) and `if wavelet_coeffs:` (loss class) into the WL-loss branch and crash on `enc_coeffs, dec_coeffs = None, None` unpacking. `modeling_wfvae2.py` collapses the pair via `extra_output = (enc, dec) if enc is not None else None`; don't revert.

## Documentation Sync

When upgrading scripts, workflows, or the model architecture, keep these in sync:
- `docs/README_image_training.md` (note: may reference outdated features like `val_patch_scores` and `training_losses.csv` — the CSV is now `{exp_name}.csv`)
- `docs/QUICK_REFERENCE.md` (may reference outdated paths and old config defaults)
- `architecture_diagram.html` and `scripts/draw_architecture.py` both describe the shipped DCAE + no-ef variant only. The script outputs two PNGs at the repo root: `wfvae2_32x_dcae_noef_architecture.png` (full architecture) and `wfvae2_32x_dcae_noef_blocks_detail.png` (WFDownBlock + WFUpBlock detail). Regenerate via `python scripts/draw_architecture.py`. Classic / paper-form ef diagrams are not drawn; the model code still supports both (for A/B) but the visual reference only shows what this branch actually trains.

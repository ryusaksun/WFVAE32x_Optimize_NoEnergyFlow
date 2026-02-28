import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, DistributedSampler, Subset
import argparse
import logging
import tqdm
from itertools import chain
from collections import deque
from contextlib import nullcontext
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from causalimagevae.model import *
from causalimagevae.model.ema_model import EMA
from causalimagevae.model.losses.discriminator import weights_init
from causalimagevae.dataset.ddp_sampler import CustomDistributedSampler
from causalimagevae.dataset.image_dataset import ImageDataset, ValidImageDataset
from causalimagevae.model.utils.module_utils import resolve_str_to_obj
from torchvision.utils import make_grid, save_image

try:
    import wandb
except ImportError:
    wandb = None

try:
    import lpips
except ImportError:
    lpips = None

import csv
import signal
import sys

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def setup_logger(rank):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        f"[rank{rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    return logger

def check_unused_params(model):
    unused_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_params.append(name)
    return unused_params

def set_requires_grad_optimizer(optimizer, requires_grad):
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            param.requires_grad = requires_grad

def total_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_millions = total_params / 1e6
    return int(total_params_in_millions)

def get_exp_name(args):
    return f"{args.exp_name}-lr{args.lr:.2e}-bs{args.batch_size}-rs{args.resolution}"

def set_train(modules):
    for module in modules:
        module.train()

def set_eval(modules):
    for module in modules:
        module.eval()

def set_modules_requires_grad(modules, requires_grad):
    for module in modules:
        module.requires_grad_(requires_grad)

def save_checkpoint(
    epoch,
    current_step,
    optimizer_state,
    state_dict,
    scaler_state,
    sampler_state,
    checkpoint_dir,
    filename="checkpoint.ckpt",
    ema_state_dict=None,
    disc_refresh_state=None,
):
    filepath = checkpoint_dir / Path(filename)
    data = {
        "epoch": epoch,
        "current_step": current_step,
        "optimizer_state": optimizer_state,
        "state_dict": state_dict,
        "ema_state_dict": ema_state_dict or {},
        "scaler_state": scaler_state,
        "sampler_state": sampler_state,
    }
    if disc_refresh_state is not None:
        data["disc_refresh_state"] = disc_refresh_state
    torch.save(data, filepath)
    return filepath

def _extract_patch_maps(logits):
    logits = logits.detach().float().cpu()
    if logits.dim() == 4:
        if logits.shape[1] == 1:
            return logits[:, 0]
        return logits.mean(dim=1)
    if logits.dim() == 3:
        return logits
    raise ValueError(f"Unexpected PatchGAN logits shape: {tuple(logits.shape)}")

def _safe_file_stem(file_name, fallback):
    stem = Path(file_name).stem if file_name else fallback
    cleaned = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in stem).strip("._")
    return cleaned or fallback

def _resolve_sample_image_path(val_dataset, sample_idx):
    base_dataset = val_dataset.dataset if isinstance(val_dataset, Subset) else val_dataset
    if not hasattr(base_dataset, "samples"):
        return None

    samples = base_dataset.samples
    if sample_idx < 0 or sample_idx >= len(samples):
        return None

    sample = samples[sample_idx]
    if isinstance(sample, str):
        return sample

    if isinstance(sample, dict):
        image_path = sample.get("image_path", sample.get("path", sample.get("target", "")))
        if image_path and (not os.path.isabs(image_path)):
            base_dir = getattr(base_dataset, "base_dir", "")
            if base_dir:
                image_path = os.path.join(base_dir, image_path)
        return image_path or None

    return None

def _resize_and_center_crop(image, resolution):
    bilinear = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    width, height = image.size
    short_side = min(width, height)
    if short_side <= 0:
        raise ValueError("Invalid image size.")

    scale = float(resolution) / float(short_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = image.resize((new_width, new_height), resample=bilinear)

    left = max(0, (new_width - resolution) // 2)
    top = max(0, (new_height - resolution) // 2)
    return image.crop((left, top, left + resolution, top + resolution))

def _load_aligned_image(val_dataset, sample_idx, eval_resolution):
    image_path = _resolve_sample_image_path(val_dataset, sample_idx)
    if image_path is None:
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        image = _resize_and_center_crop(image, eval_resolution)
        return np.asarray(image, dtype=np.uint8)
    except Exception:
        return None

def _tensor_image_to_uint8(image_tensor):
    if not torch.is_tensor(image_tensor):
        return None

    image = image_tensor.detach().float().cpu().clamp(0, 1)
    if image.dim() != 3:
        return None
    if image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0).contiguous()
    if image.shape[-1] == 1:
        image = image.repeat(1, 1, 3)
    if image.shape[-1] != 3:
        return None

    return (image.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)

def _resize_patch_map(score_map, out_height, out_width):
    h, w = score_map.shape
    row_idx = (np.arange(out_height) * h / out_height).astype(int).clip(0, h - 1)
    col_idx = (np.arange(out_width) * w / out_width).astype(int).clip(0, w - 1)
    return score_map[row_idx[:, None], col_idx[None, :]]

def _colorize_score_map(score_map):
    score_map = np.clip(score_map, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * score_map - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * score_map - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * score_map - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

def _draw_patch_grid(image_rgb, grid_h, grid_w, color):
    image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image)
    height, width = image_rgb.shape[:2]

    for col in range(1, grid_w):
        x = int(round(col * width / grid_w))
        draw.line([(x, 0), (x, height)], fill=color, width=1)
    for row in range(1, grid_h):
        y = int(round(row * height / grid_h))
        draw.line([(0, y), (width, y)], fill=color, width=1)
    return np.asarray(image, dtype=np.uint8)

def _make_colorbar(height, vmin, vmax, bar_width=30, total_width=90):
    """生成垂直 colorbar，标注 vmin/vmax"""
    canvas = np.full((height, total_width, 3), 255, dtype=np.uint8)
    margin = 10
    bar_height = height - 2 * margin

    # 渐变色条 (top=vmax, bottom=vmin)
    gradient = np.linspace(1.0, 0.0, bar_height).reshape(-1, 1)
    gradient = np.repeat(gradient, bar_width, axis=1)
    colored = _colorize_score_map(gradient)
    canvas[margin:margin + bar_height, 5:5 + bar_width] = colored

    # 用 PIL 写文字
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    text_x = 5 + bar_width + 3
    draw.text((text_x, margin - 2), f"{vmax:.3f}", fill=(0, 0, 0))
    draw.text((text_x, margin + bar_height - 10), f"{vmin:.3f}", fill=(0, 0, 0))

    return np.asarray(img, dtype=np.uint8)

def _build_patch_score_panel(base_image, score_map):
    score_map = np.asarray(score_map, dtype=np.float32)
    grid_h, grid_w = score_map.shape
    height, width = base_image.shape[:2]

    # Per-image normalization
    vmin, vmax = float(score_map.min()), float(score_map.max())
    if vmax - vmin < 1e-6:
        normalized = np.full_like(score_map, 0.5)
    else:
        normalized = (score_map - vmin) / (vmax - vmin)

    upsampled = _resize_patch_map(normalized, height, width)
    heatmap = _colorize_score_map(upsampled)
    overlay = (
        0.55 * base_image.astype(np.float32) + 0.45 * heatmap.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    # 只在 cell 足够大时画网格
    cell_size = min(width / grid_w, height / grid_h)
    if cell_size >= 32:
        original_grid = _draw_patch_grid(base_image.copy(), grid_h, grid_w, color=(180, 180, 180))
        heatmap_grid = _draw_patch_grid(heatmap, grid_h, grid_w, color=(255, 255, 255))
        overlay_grid = _draw_patch_grid(overlay, grid_h, grid_w, color=(255, 255, 255))
    else:
        original_grid = base_image.copy()
        heatmap_grid = heatmap
        overlay_grid = overlay

    # 生成 colorbar
    colorbar = _make_colorbar(height, vmin, vmax)

    return np.concatenate([original_grid, heatmap_grid, overlay_grid, colorbar], axis=1)

def _save_patch_score_visualizations(
    ordered_records,
    output_dir,
    val_dataset,
    eval_resolution,
    vis_max_samples,
    selected_sample_indices=None,
    recon_image_by_sample_idx=None,
):
    if vis_max_samples is not None and vis_max_samples <= 0:
        return
    if len(ordered_records) == 0:
        return

    if eval_resolution is None or eval_resolution <= 0:
        first_map = np.asarray(ordered_records[0]["real_sigmoid"], dtype=np.float32)
        eval_resolution = int(max(first_map.shape))

    real_dir = output_dir / "patch_vis" / "real"
    recon_dir = output_dir / "patch_vis" / "recon"
    real_dir.mkdir(exist_ok=True, parents=True)
    recon_dir.mkdir(exist_ok=True, parents=True)

    selected_records = ordered_records
    if selected_sample_indices is not None:
        sample_idx_to_record = {
            int(record["sample_idx"]): record for record in ordered_records
        }
        selected_records = [
            sample_idx_to_record[sample_idx]
            for sample_idx in selected_sample_indices
            if sample_idx in sample_idx_to_record
        ]

    max_samples = (
        len(selected_records)
        if vis_max_samples is None
        else min(len(selected_records), int(vis_max_samples))
    )
    selected_records = selected_records[:max_samples]

    for row_id, record in enumerate(selected_records):
        sample_idx = int(record["sample_idx"])
        file_name = record.get("file_name", "")
        fallback_name = f"sample_{sample_idx:08d}"
        stem = _safe_file_stem(file_name, fallback_name)
        image_name = f"{row_id:05d}_{sample_idx:08d}_{stem}.png"

        real_base_image = _load_aligned_image(val_dataset, sample_idx, eval_resolution) if val_dataset is not None else None
        if real_base_image is None:
            real_base_image = np.full((eval_resolution, eval_resolution, 3), 127, dtype=np.uint8)

        recon_base_image = None
        if recon_image_by_sample_idx is not None:
            recon_base_image = recon_image_by_sample_idx.get(sample_idx)
        if recon_base_image is None:
            recon_base_image = real_base_image

        real_panel = _build_patch_score_panel(real_base_image, record["real_sigmoid"])
        recon_panel = _build_patch_score_panel(recon_base_image, record["recon_sigmoid"])

        Image.fromarray(real_panel).save(real_dir / image_name)
        Image.fromarray(recon_panel).save(recon_dir / image_name)

def save_patch_scores(
    patch_records,
    output_dir,
    val_dataset=None,
    eval_resolution=256,
    vis_max_samples=None,
    selected_sample_indices=None,
    recon_image_by_sample_idx=None,
):
    if len(patch_records) == 0:
        return

    output_dir.mkdir(exist_ok=True, parents=True)
    ordered_records = sorted(patch_records, key=lambda x: int(x["sample_idx"]))

    summary_path = output_dir / "summary.csv"
    fieldnames = [
        "sample_idx", "file_name", "row_id",
        "real_logits_mean", "real_logits_std", "real_logits_min", "real_logits_max",
        "recon_logits_mean", "recon_logits_std", "recon_logits_min", "recon_logits_max",
        "real_sigmoid_mean", "real_sigmoid_std", "real_sigmoid_min", "real_sigmoid_max",
        "recon_sigmoid_mean", "recon_sigmoid_std", "recon_sigmoid_min", "recon_sigmoid_max",
    ]
    with open(summary_path, "w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        for row_id, record in enumerate(ordered_records):
            real_logits = np.asarray(record["real_logits"], dtype=np.float32)
            recon_logits = np.asarray(record["recon_logits"], dtype=np.float32)
            real_sigmoid = np.asarray(record["real_sigmoid"], dtype=np.float32)
            recon_sigmoid = np.asarray(record["recon_sigmoid"], dtype=np.float32)

            writer.writerow({
                "sample_idx": int(record["sample_idx"]),
                "file_name": record.get("file_name", ""),
                "row_id": row_id,
                "real_logits_mean": float(real_logits.mean()),
                "real_logits_std": float(real_logits.std()),
                "real_logits_min": float(real_logits.min()),
                "real_logits_max": float(real_logits.max()),
                "recon_logits_mean": float(recon_logits.mean()),
                "recon_logits_std": float(recon_logits.std()),
                "recon_logits_min": float(recon_logits.min()),
                "recon_logits_max": float(recon_logits.max()),
                "real_sigmoid_mean": float(real_sigmoid.mean()),
                "real_sigmoid_std": float(real_sigmoid.std()),
                "real_sigmoid_min": float(real_sigmoid.min()),
                "real_sigmoid_max": float(real_sigmoid.max()),
                "recon_sigmoid_mean": float(recon_sigmoid.mean()),
                "recon_sigmoid_std": float(recon_sigmoid.std()),
                "recon_sigmoid_min": float(recon_sigmoid.min()),
                "recon_sigmoid_max": float(recon_sigmoid.max()),
            })

    _save_patch_score_visualizations(
        ordered_records=ordered_records,
        output_dir=output_dir,
        val_dataset=val_dataset,
        eval_resolution=int(eval_resolution) if eval_resolution is not None else None,
        vis_max_samples=vis_max_samples,
        selected_sample_indices=selected_sample_indices,
        recon_image_by_sample_idx=recon_image_by_sample_idx,
    )

def log_patch_scores_to_wandb(patch_records, step, prefix, use_wandb=True):
    if (not use_wandb) or wandb is None or len(patch_records) == 0:
        return

    ordered_records = sorted(patch_records, key=lambda x: int(x["sample_idx"]))
    real_logits_flat = np.concatenate(
        [np.asarray(record["real_logits"], dtype=np.float32).reshape(-1) for record in ordered_records]
    )
    recon_logits_flat = np.concatenate(
        [np.asarray(record["recon_logits"], dtype=np.float32).reshape(-1) for record in ordered_records]
    )
    real_sigmoid_flat = np.concatenate(
        [np.asarray(record["real_sigmoid"], dtype=np.float32).reshape(-1) for record in ordered_records]
    )
    recon_sigmoid_flat = np.concatenate(
        [np.asarray(record["recon_sigmoid"], dtype=np.float32).reshape(-1) for record in ordered_records]
    )

    wandb.log(
        {
            f"{prefix}/patch_real_logits_mean": float(real_logits_flat.mean()),
            f"{prefix}/patch_recon_logits_mean": float(recon_logits_flat.mean()),
            f"{prefix}/patch_real_sigmoid_mean": float(real_sigmoid_flat.mean()),
            f"{prefix}/patch_recon_sigmoid_mean": float(recon_sigmoid_flat.mean()),
            f"{prefix}/patch_real_logits_hist": wandb.Histogram(real_logits_flat),
            f"{prefix}/patch_recon_logits_hist": wandb.Histogram(recon_logits_flat),
            f"{prefix}/patch_real_sigmoid_hist": wandb.Histogram(real_sigmoid_flat),
            f"{prefix}/patch_recon_sigmoid_hist": wandb.Histogram(recon_sigmoid_flat),
        },
        step=step,
    )

def valid(global_rank, rank, model, discriminator, val_dataloader, precision, args, lpips_model=None):

    bar = None
    if global_rank == 0:
        bar = tqdm.tqdm(total=len(val_dataloader), desc="Validation...")

    psnr_list = []
    lpips_list = []
    orig_images = []
    recon_images = []
    logged_sample_indices = []
    patch_records = []
    num_image_log = args.eval_num_image_log

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs = batch["image"].to(rank)
            sample_indices = batch.get("index")
            if sample_indices is None:
                start_idx = batch_idx * inputs.shape[0]
                sample_indices = list(range(start_idx, start_idx + inputs.shape[0]))
            elif torch.is_tensor(sample_indices):
                sample_indices = sample_indices.tolist()
            else:
                sample_indices = [int(i) for i in sample_indices]

            file_names = batch.get("file_name")
            if file_names is None:
                file_names = [f"sample_{sample_indices[i]}" for i in range(inputs.shape[0])]
            else:
                file_names = list(file_names)

            with torch.amp.autocast("cuda", dtype=precision):
                output = model(inputs)
                image_recon = output.sample
                logits_real = discriminator(inputs)
                logits_recon = discriminator(image_recon)

            real_logits_maps = _extract_patch_maps(logits_real)
            recon_logits_maps = _extract_patch_maps(logits_recon)
            real_sigmoid_maps = torch.sigmoid(real_logits_maps)
            recon_sigmoid_maps = torch.sigmoid(recon_logits_maps)

            for i in range(len(image_recon)):
                patch_records.append({
                    "sample_idx": int(sample_indices[i]),
                    "file_name": file_names[i] if i < len(file_names) else f"sample_{sample_indices[i]}",
                    "real_logits": real_logits_maps[i].numpy(),
                    "recon_logits": recon_logits_maps[i].numpy(),
                    "real_sigmoid": real_sigmoid_maps[i].numpy(),
                    "recon_sigmoid": recon_sigmoid_maps[i].numpy(),
                })

            # Collect images for logging
            if global_rank == 0:
                for i in range(len(image_recon)):
                    if num_image_log <= 0:
                        break
                    # Normalize from [-1, 1] to [0, 1], convert to float32 for numpy compatibility
                    img_orig = ((inputs[i] + 1.0) / 2.0).float().cpu().clamp(0, 1)
                    img_recon = ((image_recon[i] + 1.0) / 2.0).float().cpu().clamp(0, 1)
                    orig_images.append(img_orig)
                    recon_images.append(img_recon)
                    logged_sample_indices.append(int(sample_indices[i]))
                    num_image_log -= 1

            # Calculate PSNR (data range [-1, 1], so MAX=2)
            mse = torch.mean(torch.square(inputs - image_recon), dim=(1, 2, 3))
            psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
            psnr = psnr.mean().detach().cpu().item()

            # Calculate LPIPS
            if args.eval_lpips:
                lpips_score = (
                    lpips_model.forward(inputs, image_recon)
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )
                lpips_list.append(lpips_score)

            psnr_list.append(psnr)

            if global_rank == 0:
                bar.update()
            # Release gpus memory
            torch.cuda.empty_cache()
    return psnr_list, lpips_list, orig_images, recon_images, patch_records, logged_sample_indices

def gather_valid_result(psnr_list, lpips_list, orig_images, recon_images, patch_records, rank, world_size):
    gathered_psnr_list = [None for _ in range(world_size)]
    gathered_lpips_list = [None for _ in range(world_size)]
    gathered_orig_images = [None for _ in range(world_size)]
    gathered_recon_images = [None for _ in range(world_size)]
    gathered_patch_records = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_psnr_list, psnr_list)
    dist.all_gather_object(gathered_lpips_list, lpips_list)
    dist.all_gather_object(gathered_orig_images, orig_images)
    dist.all_gather_object(gathered_recon_images, recon_images)
    dist.all_gather_object(gathered_patch_records, patch_records)

    all_psnr = list(chain(*gathered_psnr_list))
    all_lpips = list(chain(*gathered_lpips_list))
    all_patch_records = list(chain(*gathered_patch_records))

    patch_record_dict = {}
    for record in all_patch_records:
        sample_idx = int(record["sample_idx"])
        if sample_idx not in patch_record_dict:
            patch_record_dict[sample_idx] = record
    deduplicated_patch_records = [
        patch_record_dict[sample_idx] for sample_idx in sorted(patch_record_dict.keys())
    ]

    return (
        float(np.mean(all_psnr)) if len(all_psnr) > 0 else float("nan"),
        float(np.mean(all_lpips)) if len(all_lpips) > 0 else float("nan"),
        list(chain(*gathered_orig_images)),
        list(chain(*gathered_recon_images)),
        deduplicated_patch_records,
    )

def _to_csv_scalar(value):
    if value is None or value == "":
        return ""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return ""
        return float(value.detach().mean().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""

def _coerce_numeric_series(series, pd):
    numeric_series = pd.to_numeric(series, errors="coerce")
    need_fallback = numeric_series.isna() & series.notna()
    if need_fallback.any():
        extracted = (
            series[need_fallback]
            .astype(str)
            .str.extract(r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", expand=False)
        )
        numeric_series.loc[need_fallback] = pd.to_numeric(extracted, errors="coerce")
    return numeric_series

def plot_training_curves(csv_path, output_path, disc_start=None, smoothing_window=50):
    """
    Plots training curves from CSV log file with dynamic multi-subplot layout.

    Args:
        csv_path: Path to the CSV file containing training logs
        output_path: Path to save the output plot
        disc_start: Step when discriminator training starts (draws vertical line)
        smoothing_window: Window size for moving average smoothing (default: 50)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for servers
        import matplotlib.pyplot as plt
        import pandas as pd
        from scipy.ndimage import uniform_filter1d
    except ImportError as e:
        print(f"Warning: Could not import plotting libraries: {e}")
        return

    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found at {csv_path}")
        return

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Warning: Could not read CSV file: {e}")
        return

    if len(df) == 0:
        print("Warning: CSV file is empty")
        return

    # Extract discriminator refresh steps from marker rows
    disc_refresh_steps = []
    if 'generator_loss' in df.columns:
        refresh_mask = df['generator_loss'].astype(str).str.strip().isin(['DISC_REFRESH', 'DISC_AUTO_REFRESH'])
        refresh_rows = df.loc[refresh_mask, 'step']
        disc_refresh_steps = pd.to_numeric(refresh_rows, errors='coerce').dropna().tolist()

    # Define metrics to plot
    metrics = [
        ('generator_loss', 'Generator Loss', 'tab:blue'),
        ('discriminator_loss', 'Discriminator Loss', 'tab:orange'),
        ('rec_loss', 'Reconstruction Loss', 'tab:green'),
        ('perceptual_loss', 'Perceptual Loss', 'tab:brown'),
        ('kl_loss', 'KL Loss', 'tab:red'),
        ('wavelet_loss', 'Wavelet Loss', 'tab:purple'),
        ('logvar', 'Log Variance', 'teal'),
        ('nll_loss', 'NLL Loss', 'slateblue'),
        ('g_loss', 'GAN Loss (g_loss)', 'tab:cyan'),
        ('d_weight', 'Adaptive Weight (d_weight)', 'darkgoldenrod'),
        ('logits_real', 'Logits Real', 'forestgreen'),
        ('logits_fake', 'Logits Fake', 'tomato'),
        ('r1_penalty', 'R1 Penalty', 'crimson'),
        ('r1_effective_weight', 'R1 Effective Weight', 'hotpink'),
        ('nll_grads_norm', 'NLL Grad Norm', 'mediumpurple'),
        ('g_grads_norm', 'GAN Grad Norm', 'coral'),
        ('psnr', 'Validation PSNR', 'deepskyblue'),
        ('lpips', 'Validation LPIPS', 'tab:olive'),
        ('psnr_ema', 'Validation PSNR (EMA)', 'tab:pink'),
        ('lpips_ema', 'Validation LPIPS (EMA)', 'tab:gray'),
    ]

    # Create figure with dynamic subplot layout
    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle('Training Curves', fontsize=16, y=0.995)

    # Flatten axes for easier iteration
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, (metric_key, title, color) in enumerate(metrics):
        ax = axes_flat[idx]

        # Check if metric exists in dataframe
        if metric_key not in df.columns:
            ax.text(0.5, 0.5, f'{title}\n(No data)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=12)
            continue

        # Get data and coerce possible legacy string values like "tensor(...)"
        data = df[['step', metric_key]].copy()
        data['step'] = pd.to_numeric(data['step'], errors='coerce')
        data[metric_key] = _coerce_numeric_series(data[metric_key], pd)
        data = data.dropna()

        if len(data) == 0:
            ax.text(0.5, 0.5, f'{title}\n(No data)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=12)
            continue

        steps = data['step'].values
        values = data[metric_key].values

        # Sparse metrics (validation) get solid lines + markers;
        # dense metrics get transparent raw + smoothed overlay
        is_sparse = len(values) <= smoothing_window
        if is_sparse:
            ax.plot(steps, values, color=color, linewidth=1.8,
                    marker='o', markersize=4, label='Val')
        else:
            ax.plot(steps, values, alpha=0.25, color=color, linewidth=0.5, label='Raw')
            try:
                smoothed = uniform_filter1d(values, size=smoothing_window, mode='nearest')
                ax.plot(steps, smoothed, color=color, linewidth=2, label=f'Smoothed (w={smoothing_window})')
            except Exception as e:
                print(f"Warning: Could not smooth {metric_key}: {e}")

        ax.legend(loc='best', fontsize=8, framealpha=0.8)

        # Draw discriminator start line if applicable
        if disc_start is not None and disc_start > 0:
            ax.axvline(x=disc_start, color='red', linestyle='--',
                      linewidth=1, alpha=0.5)

        # Draw discriminator refresh lines
        for rs in disc_refresh_steps:
            ax.axvline(x=rs, color='darkorange', linestyle=':',
                      linewidth=1.2, alpha=0.7)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.tick_params(labelsize=9)

    # Hide unused axes when metric count is not a multiple of columns
    for idx in range(len(metrics), len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save plot
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {output_path}")
    except Exception as e:
        print(f"Warning: Could not save plot: {e}")
    finally:
        plt.close(fig)

def train(args):
    # setup logger
    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    logger = setup_logger(rank)

    # init
    ckpt_dir = Path(args.ckpt_dir) / Path(get_exp_name(args))
    if global_rank == 0:
        try:
            ckpt_dir.mkdir(exist_ok=False, parents=True)
        except:
            logger.warning(f"`{ckpt_dir}` exists!")
    dist.barrier()

    # load generator model
    model_cls = ModelRegistry.get_model(args.model_name)

    if not model_cls:
        raise ModuleNotFoundError(
            f"`{args.model_name}` not in {str(ModelRegistry._models.keys())}."
        )

    if args.pretrained_model_name_or_path is not None:
        if global_rank == 0:
            logger.warning(
                f"You are loading a checkpoint from `{args.pretrained_model_name_or_path}`."
            )
        model = model_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            low_cpu_mem_usage=False,
            device_map=None,
        )
    else:
        if global_rank == 0:
            logger.warning(f"Model will be inited randomly.")
        model = model_cls.from_config(args.model_config)

    use_wandb = not args.disable_wandb
    if use_wandb and wandb is None:
        if global_rank == 0:
            logger.warning("`wandb` 未安装，自动关闭 wandb 日志。")
        use_wandb = False

    if global_rank == 0:
        if use_wandb:
            logger.warning("Connecting to WANDB...")
            model_config = dict(**model.config)
            args_config = dict(**vars(args))
            if 'resolution' in model_config:
                del model_config['resolution']

            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "causalimagevae"),
                config=dict(**model_config, **args_config),
                name=get_exp_name(args),
            )
        else:
            logger.warning("WANDB 已禁用，仅写入本地日志与产物文件。")

    dist.barrier()

    # load discriminator model
    disc_cls = resolve_str_to_obj(args.disc_cls, append=False)
    logger.warning(
        f"disc_class: {args.disc_cls} perceptual_weight: {args.perceptual_weight}  loss_type: {args.loss_type}"
    )
    disc = disc_cls(
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        kl_weight=args.kl_weight,
        logvar_init=args.logvar_init,
        perceptual_weight=args.perceptual_weight,
        loss_type=args.loss_type,
        wavelet_weight=args.wavelet_weight,
        adaptive_weight_clamp=args.adaptive_weight_clamp,
        r1_weight=args.r1_weight,
        r1_start=args.r1_start,
        r1_warmup_steps=args.r1_warmup_steps,
        learn_logvar=args.learn_logvar,
    )

    # DDP
    model = model.to(rank)
    model = DDP(
        model, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
    )
    disc = disc.to(rank)
    disc = DDP(
        disc, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
    )

    # load dataset
    if args.use_manifest:
        from causalimagevae.dataset.manifest_dataset import ManifestImageDataset, ValidManifestImageDataset
        dataset = ManifestImageDataset(
            manifest_path=args.image_path,
            resolution=args.resolution,
        )
    else:
        dataset = ImageDataset(
            args.image_path,
            resolution=args.resolution,
            cache_file="image_cache.pkl",
            is_main_process=global_rank == 0,
        )

    ddp_sampler = CustomDistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=ddp_sampler,
        pin_memory=True,
        num_workers=args.dataset_num_worker,
    )
    if args.use_manifest:
        val_dataset = ValidManifestImageDataset(
            manifest_path=args.eval_image_path,
            resolution=args.eval_resolution,
        )
    else:
        val_dataset = ValidImageDataset(
            image_folder=args.eval_image_path,
            resolution=args.eval_resolution,
            crop_size=args.eval_resolution,
            cache_file="valid_image_cache.pkl",
            is_main_process=global_rank == 0,
        )
    if args.eval_subset_size is not None and args.eval_subset_size > 0:
        indices = range(min(args.eval_subset_size, len(val_dataset)))
    else:
        indices = range(len(val_dataset))
    val_dataset = Subset(val_dataset, indices=indices)
    val_sampler = CustomDistributedSampler(val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        sampler=val_sampler,
        pin_memory=True,
    )

    # optimizer
    modules_to_train = [module for module in model.module.get_decoder()]
    if args.freeze_encoder:
        for module in model.module.get_encoder():
            module.eval()
            module.requires_grad_(False)
        logger.info("Encoder is freezed!")
    else:
        modules_to_train += [module for module in model.module.get_encoder()]

    parameters_to_train = []
    for module in modules_to_train:
        parameters_to_train += list(filter(lambda p: p.requires_grad, module.parameters()))

    # logvar controls reconstruction loss scaling — use separate (higher) lr for fast equilibrium
    if disc.module.logvar.requires_grad:
        logvar_lr = args.logvar_lr if args.logvar_lr is not None else args.lr
        gen_param_groups = [
            {"params": parameters_to_train, "lr": args.lr},
            {"params": [disc.module.logvar], "lr": logvar_lr, "weight_decay": 0.0},
        ]
        logger.info(f"learn_logvar enabled: logvar lr={logvar_lr} (init={disc.module.logvar.item():.4f})")
    else:
        gen_param_groups = parameters_to_train

    gen_optimizer = torch.optim.AdamW(gen_param_groups, lr=args.lr, weight_decay=args.weight_decay)
    disc_lr = args.disc_lr if args.disc_lr is not None else args.lr
    disc_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, disc.module.discriminator.parameters()), lr=disc_lr, weight_decay=args.weight_decay
    )
    if disc_lr != args.lr:
        logger.info(f"TTUR enabled: generator lr={args.lr}, discriminator lr={disc_lr}")

    # AMP scaler — only needed for fp16; bfloat16/fp32 have sufficient dynamic range
    precision = torch.bfloat16
    if args.mix_precision == "fp16":
        precision = torch.float16
    elif args.mix_precision == "fp32":
        precision = torch.float32
    scaler_enabled = (precision == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled)

    # load from checkpoint
    start_epoch = 0
    current_step = 0
    if args.resume_from_checkpoint:
        if not os.path.isfile(args.resume_from_checkpoint):
            raise Exception(
                f"Make sure `{args.resume_from_checkpoint}` is a ckpt file."
            )
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.module.load_state_dict(checkpoint["state_dict"]["gen_model"], strict=False)

        # resume optimizer
        if not args.not_resume_optimizer:
            gen_optimizer.load_state_dict(checkpoint["optimizer_state"]["gen_optimizer"])

        # resume discriminator
        if args.refresh_discriminator:
            # Refresh mode: reload non-discriminator weights (logvar, perceptual_loss),
            # but reinitialize discriminator weights and optimizer from scratch.
            full_disc_state = checkpoint["state_dict"]["dics_model"]
            non_disc_keys = {k: v for k, v in full_disc_state.items() if not k.startswith("discriminator.")}
            disc.module.load_state_dict(non_disc_keys, strict=False)
            disc.module.discriminator.apply(weights_init)
            # Broadcast rank 0 weights to all ranks to keep DDP in sync
            for param in disc.module.discriminator.parameters():
                dist.broadcast(param.data, src=0)
            disc_optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, disc.module.discriminator.parameters()),
                lr=disc_lr, weight_decay=args.weight_decay,
            )
            scaler.load_state_dict(checkpoint["scaler_state"])
            logger.info("Discriminator REFRESHED at resume (weights reinitialized, optimizer reset).")
        elif not args.not_resume_discriminator:
            disc.module.load_state_dict(checkpoint["state_dict"]["dics_model"])
            if not args.not_resume_optimizer:
                disc_optimizer.load_state_dict(checkpoint["optimizer_state"]["disc_optimizer"])
                # Override disc lr if --disc_lr is explicitly set (TTUR), since checkpoint saves the old lr
                if args.disc_lr is not None:
                    for pg in disc_optimizer.param_groups:
                        pg['lr'] = disc_lr
                    logger.info(f"Overriding resumed discriminator lr to {disc_lr} (TTUR)")
            scaler.load_state_dict(checkpoint["scaler_state"])

        # resume data sampler and training progress
        if not args.not_resume_training_process:
            ddp_sampler.load_state_dict(checkpoint["sampler_state"])
            start_epoch = checkpoint["sampler_state"]["epoch"]
            current_step = checkpoint["current_step"]
            logger.info(
                f"Checkpoint loaded from {args.resume_from_checkpoint}, starting from epoch {start_epoch} step {current_step}"
            )
        else:
            logger.info(
                f"Checkpoint weights loaded from {args.resume_from_checkpoint}, training process reset (epoch=0, step=0)"
            )

    if args.ema:
        logger.warning(f"Start with EMA. EMA decay = {args.ema_decay}.")
        ema = EMA(model, args.ema_decay)
        ema.register()
        # Restore EMA shadow weights from checkpoint
        if args.resume_from_checkpoint and "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"]:
            ema.shadow = checkpoint["ema_state_dict"]
            logger.info("EMA shadow weights restored from checkpoint.")

    logger.info("Prepared!")

    # Initialize CSV logger
    csv_file = None
    csv_writer = None
    csv_supports_ema_columns = False
    csv_warned_ema_legacy = False
    csv_name = f"{args.exp_name}.csv"
    if global_rank == 0:
        csv_path = ckpt_dir / csv_name
        fieldnames = [
            "step", "generator_loss", "discriminator_loss",
            "rec_loss", "perceptual_loss", "kl_loss", "wavelet_loss",
            "logvar", "nll_loss",
            "g_loss", "d_weight",
            "logits_real", "logits_fake", "r1_penalty", "r1_effective_weight",
            "nll_grads_norm", "g_grads_norm",
            "psnr", "lpips", "psnr_ema", "lpips_ema",
        ]

        # Check if resuming from checkpoint (fall back to old filename)
        if args.resume_from_checkpoint and not csv_path.exists():
            legacy_csv = ckpt_dir / "training_losses.csv"
            if legacy_csv.exists() and legacy_csv.stat().st_size > 0:
                logger.info(f"Renaming legacy CSV: {legacy_csv} -> {csv_path}")
                legacy_csv.rename(csv_path)
        resume_csv = (
            args.resume_from_checkpoint
            and csv_path.exists()
            and csv_path.stat().st_size > 0
        )
        if resume_csv:
            logger.info(f"Resuming CSV logging to {csv_path}")
            try:
                with open(csv_path, "r", newline="") as existing_csv_file:
                    existing_header = next(csv.reader(existing_csv_file), None)
                if existing_header:
                    # Merge: keep existing columns and append any new ones
                    missing_cols = [c for c in fieldnames if c not in existing_header]
                    if missing_cols:
                        logger.info(f"Upgrading CSV: adding new columns {missing_cols}")
                        fieldnames = existing_header + missing_cols
                        # Rewrite file with updated header (old data rows unchanged)
                        with open(csv_path, "r", newline="") as f:
                            old_lines = f.readlines()
                        with open(csv_path, "w", newline="") as f:
                            f.write(",".join(fieldnames) + "\n")
                            f.writelines(old_lines[1:])  # skip old header
                    else:
                        fieldnames = existing_header
            except Exception as e:
                logger.warning(f"Failed to read CSV header from `{csv_path}`: {e}")

            csv_supports_ema_columns = all(
                col in fieldnames for col in ("psnr_ema", "lpips_ema")
            )
            if not csv_supports_ema_columns:
                logger.warning(
                    "Resumed CSV does not contain `psnr_ema/lpips_ema` columns; "
                    "EMA validation metrics will be skipped in CSV to avoid mixing."
                )
            csv_file = open(csv_path, "a", newline="")
            csv_writer = csv.DictWriter(
                csv_file, fieldnames=fieldnames, extrasaction="ignore"
            )
        else:
            logger.info(f"Starting new CSV log at {csv_path}")
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.DictWriter(
                csv_file, fieldnames=fieldnames, extrasaction="ignore"
            )
            csv_writer.writeheader()
            csv_supports_ema_columns = True

    # Setup signal handlers for graceful interruption
    def signal_handler(signum, frame):
        """Handle Ctrl+C and kill signals gracefully"""
        if global_rank == 0:
            logger.warning(f"\nReceived signal {signum}. Gracefully shutting down...")

            # Close CSV file
            if csv_file is not None:
                try:
                    csv_file.flush()
                    csv_file.close()
                    logger.info("CSV file closed successfully")
                except Exception as e:
                    logger.error(f"Error closing CSV file: {e}")

            # Generate final plot with '_interrupted' suffix
            if not args.disable_plot:
                try:
                    csv_path = ckpt_dir / csv_name
                    plot_path = ckpt_dir / f"training_curves_interrupted_step{current_step}.png"
                    logger.info(f"Generating interrupted training plot at {plot_path}")
                    plot_training_curves(
                        csv_path=csv_path,
                        output_path=plot_path,
                        disc_start=args.disc_start,
                        smoothing_window=50
                    )
                except Exception as e:
                    logger.error(f"Error generating interrupted plot: {e}")

            logger.info("Shutdown complete. Exiting...")

        # Cleanup distributed processes
        try:
            dist.destroy_process_group()
        except:
            pass

        sys.exit(0)

    # Register signal handlers on all ranks so non-rank-0 processes also exit cleanly
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command

    dist.barrier()
    if global_rank == 0:
        logger.info(f"Generator:\t\t{total_params(model.module)}M")
        logger.info(f"\t- Encoder:\t{total_params(model.module.encoder):d}M")
        logger.info(f"\t- Decoder:\t{total_params(model.module.decoder):d}M")
        logger.info(f"Discriminator:\t{total_params(disc.module):d}M")
        logger.info(f"Precision is set to: {args.mix_precision}!")
        logger.info("Start training!")

    # Create LPIPS model once for validation (avoid recreating DDP wrapper each time)
    val_lpips_model = None
    if args.eval_lpips:
        if lpips is None:
            raise ImportError("lpips is required when --eval_lpips is enabled. Install with: pip install lpips")
        val_lpips_model = lpips.LPIPS(net="alex", spatial=True)
        val_lpips_model.to(rank)
        val_lpips_model = DDP(val_lpips_model, device_ids=[rank])
        val_lpips_model.requires_grad_(False)
        val_lpips_model.eval()

    # training bar
    bar_desc = "Epoch: {current_epoch}, Loss: {loss}"
    bar = None
    if global_rank == 0:
        max_steps = (
            args.epochs * len(dataloader) if args.max_steps is None else args.max_steps
        )
        bar = tqdm.tqdm(total=max_steps, desc=bar_desc.format(current_epoch=0, loss=0))
        bar.update(current_step)
        logger.warning("Training Details: ")
        logger.warning(f" Max steps: {max_steps}")
        logger.warning(f" Dataset Samples: {len(dataset)}")
        logger.warning(
            f" Total Batch Size: {args.batch_size} * {os.environ['WORLD_SIZE']}"
        )
    dist.barrier()

    num_epochs = args.epochs
    stop_training = False
    last_gen_csv_logged_step = -args.csv_log_steps
    last_disc_csv_logged_step = -args.csv_log_steps

    # Discriminator refresh tracking
    last_disc_refresh_step = None
    disc_loss_buffer = deque(maxlen=args.disc_stale_window)
    if args.refresh_discriminator and args.resume_from_checkpoint:
        last_disc_refresh_step = current_step
        if global_rank == 0:
            logger.info(f"Discriminator warmup active for {args.disc_refresh_warmup} steps starting at step {current_step}.")
    elif args.resume_from_checkpoint and args.disc_auto_refresh and "disc_refresh_state" in checkpoint:
        drs = checkpoint["disc_refresh_state"]
        last_disc_refresh_step = drs.get("last_disc_refresh_step")
        saved_buffer = drs.get("disc_loss_buffer", [])
        disc_loss_buffer.extend(saved_buffer)
        if global_rank == 0:
            logger.info(f"Disc refresh state restored: last_refresh_step={last_disc_refresh_step}, buffer_len={len(disc_loss_buffer)}")

    def update_bar(bar):
        if global_rank == 0:
            bar.desc = bar_desc.format(current_epoch=epoch, loss=f"-")
            bar.update()

    # training Loop
    for epoch in range(start_epoch, num_epochs):
        set_train(modules_to_train)
        ddp_sampler.set_epoch(epoch)  # Shuffle data at every epoch

        for batch_idx, batch in enumerate(dataloader):
            if args.max_steps is not None and current_step >= args.max_steps:
                stop_training = True
                break

            inputs = batch["image"].to(rank)

            # --- Periodic discriminator refresh ---
            if (
                args.disc_refresh_every > 0
                and current_step > 0
                and current_step % args.disc_refresh_every == 0
                and (last_disc_refresh_step is None or current_step != last_disc_refresh_step)
            ):
                disc.module.discriminator.apply(weights_init)
                # Broadcast rank 0 weights to all ranks to keep DDP in sync
                for param in disc.module.discriminator.parameters():
                    dist.broadcast(param.data, src=0)
                disc_optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, disc.module.discriminator.parameters()),
                    lr=disc_lr, weight_decay=args.weight_decay,
                )
                last_disc_refresh_step = current_step
                if global_rank == 0:
                    logger.info(f"[Disc Refresh] Periodic refresh at step {current_step}, warmup for {args.disc_refresh_warmup} steps.")
                    if csv_writer is not None:
                        try:
                            csv_writer.writerow({
                                "step": current_step,
                                "generator_loss": "DISC_REFRESH",
                                "discriminator_loss": "", "rec_loss": "",
                                "perceptual_loss": "", "kl_loss": "",
                                "wavelet_loss": "", "logvar": "", "nll_loss": "",
                                "g_loss": "", "d_weight": "",
                                "logits_real": "", "logits_fake": "", "r1_penalty": "", "r1_effective_weight": "",
                                "nll_grads_norm": "", "g_grads_norm": "",
                                "psnr": "", "lpips": "",
                                "psnr_ema": "", "lpips_ema": "",
                            })
                            csv_file.flush()
                        except Exception:
                            pass

            # --- Auto discriminator refresh (staleness detection) ---
            if (
                args.disc_auto_refresh
                and current_step >= disc.module.discriminator_iter_start
                and len(disc_loss_buffer) >= args.disc_stale_window
            ):
                cooldown_ok = (
                    last_disc_refresh_step is None
                    or (current_step - last_disc_refresh_step) >= args.disc_auto_refresh_cooldown
                )
                if cooldown_ok:
                    import statistics
                    buf_mean = statistics.mean(disc_loss_buffer)
                    buf_std = statistics.pstdev(disc_loss_buffer)
                    if buf_mean > args.disc_stale_loss_threshold and buf_std < args.disc_stale_std_threshold:
                        disc.module.discriminator.apply(weights_init)
                        # Broadcast rank 0 weights to all ranks to keep DDP in sync
                        for param in disc.module.discriminator.parameters():
                            dist.broadcast(param.data, src=0)
                        disc_optimizer = torch.optim.AdamW(
                            filter(lambda p: p.requires_grad, disc.module.discriminator.parameters()),
                            lr=disc_lr, weight_decay=args.weight_decay,
                        )
                        last_disc_refresh_step = current_step
                        disc_loss_buffer.clear()
                        if global_rank == 0:
                            logger.info(
                                f"[Auto Disc Refresh] Staleness detected at step {current_step}: "
                                f"mean={buf_mean:.4f}, std={buf_std:.4f}. "
                                f"Warmup for {args.disc_refresh_warmup} steps."
                            )
                            if csv_writer is not None:
                                try:
                                    csv_writer.writerow({
                                        "step": current_step,
                                        "generator_loss": "DISC_AUTO_REFRESH",
                                        "discriminator_loss": "", "rec_loss": "",
                                        "perceptual_loss": "", "kl_loss": "",
                                        "wavelet_loss": "", "logvar": "", "nll_loss": "",
                                        "g_loss": "", "d_weight": "",
                                        "logits_real": "", "logits_fake": "", "r1_penalty": "", "r1_effective_weight": "",
                                        "nll_grads_norm": "", "g_grads_norm": "",
                                        "psnr": "", "lpips": "",
                                        "psnr_ema": "", "lpips_ema": "",
                                    })
                                    csv_file.flush()
                                except Exception:
                                    pass

            # --- Select generator or discriminator step ---
            in_disc_warmup = (
                last_disc_refresh_step is not None
                and (current_step - last_disc_refresh_step) < args.disc_refresh_warmup
            )

            if in_disc_warmup:
                # Warmup: only train discriminator after refresh
                set_modules_requires_grad(modules_to_train, False)
                disc.module.discriminator.requires_grad_(True)
                step_gen = False
                step_dis = True
            elif (
                current_step % 2 == 1
                and current_step >= disc.module.discriminator_iter_start
            ):
                set_modules_requires_grad(modules_to_train, False)
                disc.module.discriminator.requires_grad_(True)
                step_gen = False
                step_dis = True
            else:
                set_modules_requires_grad(modules_to_train, True)
                disc.module.discriminator.requires_grad_(False)
                step_gen = True
                step_dis = False

            assert (
                step_gen or step_dis
            ), "You should backward either Gen. or Dis. in a step."

            # forward (use no_grad for disc steps to save memory)
            fwd_ctx = torch.no_grad() if step_dis else nullcontext()
            with fwd_ctx, torch.amp.autocast('cuda', dtype=precision):
                outputs = model(inputs)
                recon = outputs.sample
                posterior = outputs.latent_dist
                wavelet_coeffs = None
                if outputs.extra_output is not None and args.wavelet_loss:
                    wavelet_coeffs = outputs.extra_output

            # generator loss
            if step_gen:
                with torch.amp.autocast('cuda', dtype=precision):
                    # Use disc.module directly to bypass DDP (no gradient sync needed for disc in gen step)
                    g_loss, g_log = disc.module(
                        inputs,
                        recon,
                        posterior,
                        optimizer_idx=0, # 0 - generator
                        global_step=current_step,
                        last_layer=model.module.get_last_layer(),
                        wavelet_coeffs=wavelet_coeffs,
                        split="train",
                    )
                gen_optimizer.zero_grad()
                scaler.scale(g_loss).backward()
                scaler.unscale_(gen_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(gen_optimizer)
                scaler.update()

                # update ema
                if args.ema:
                    ema.update()

                # log to wandb
                if use_wandb and global_rank == 0 and current_step % args.log_steps == 0:
                    wandb.log(
                        {"train/generator_loss": g_loss.item()}, step=current_step
                    )
                    wandb.log(
                        {"train/rec_loss": g_log['train/rec_loss']}, step=current_step
                    )
                    wandb.log(
                        {"train/latents_std": posterior.sample().std().item()}, step=current_step
                    )

                # Log generator-related metrics to CSV
                if (
                    csv_writer is not None
                    and (current_step - last_gen_csv_logged_step) >= args.csv_log_steps
                ):
                    try:
                        csv_writer.writerow({
                            "step": current_step,
                            "generator_loss": _to_csv_scalar(g_loss),
                            "discriminator_loss": "",
                            "rec_loss": _to_csv_scalar(g_log.get('train/rec_loss')),
                            "perceptual_loss": _to_csv_scalar(g_log.get('train/p_loss')),
                            "kl_loss": _to_csv_scalar(g_log.get('train/kl_loss')),
                            "wavelet_loss": _to_csv_scalar(g_log.get('train/wl_loss')),
                            "logvar": _to_csv_scalar(g_log.get('train/logvar')),
                            "nll_loss": _to_csv_scalar(g_log.get('train/nll_loss')),
                            "g_loss": _to_csv_scalar(g_log.get('train/g_loss')),
                            "d_weight": _to_csv_scalar(g_log.get('train/d_weight')),
                            "logits_real": "",
                            "logits_fake": "",
                            "r1_penalty": "",
                            "r1_effective_weight": "",
                            "nll_grads_norm": _to_csv_scalar(g_log.get('train/nll_grads_norm')),
                            "g_grads_norm": _to_csv_scalar(g_log.get('train/g_grads_norm')),
                            "psnr": "",
                            "lpips": "",
                            "psnr_ema": "",
                            "lpips_ema": "",
                        })
                        csv_file.flush()
                        last_gen_csv_logged_step = current_step
                    except Exception as e:
                        logger.error(f"Failed to write generator metrics to CSV: {e}")

            # discriminator loss
            if step_dis:
                with torch.amp.autocast('cuda', dtype=precision):
                    d_loss, d_log = disc(
                        inputs,
                        recon,
                        posterior,
                        optimizer_idx=1,
                        global_step=current_step,
                        last_layer=None,
                        split="train",
                    )
                disc_optimizer.zero_grad()
                scaler.scale(d_loss).backward()
                scaler.unscale_(disc_optimizer)
                torch.nn.utils.clip_grad_norm_(disc.module.discriminator.parameters(), args.clip_grad_norm)
                scaler.step(disc_optimizer)
                scaler.update()
                if args.disc_auto_refresh:
                    # Use globally averaged loss so staleness detection is consistent across ranks
                    d_loss_avg = d_loss.detach().clone()
                    dist.all_reduce(d_loss_avg, op=dist.ReduceOp.AVG)
                    disc_loss_buffer.append(d_loss_avg.item())
                if use_wandb and global_rank == 0 and current_step % args.log_steps == 0:
                    wandb.log(
                        {"train/discriminator_loss": d_loss.item()}, step=current_step
                    )

                # Log discriminator-related metrics to CSV
                if (
                    csv_writer is not None
                    and (current_step - last_disc_csv_logged_step) >= args.csv_log_steps
                ):
                    try:
                        csv_writer.writerow({
                            "step": current_step,
                            "generator_loss": "",
                            "discriminator_loss": _to_csv_scalar(d_log.get('train/disc_loss', d_loss)),
                            "rec_loss": "",
                            "perceptual_loss": "",
                            "kl_loss": "",
                            "wavelet_loss": "",
                            "logvar": "",
                            "nll_loss": "",
                            "g_loss": "",
                            "d_weight": "",
                            "logits_real": _to_csv_scalar(d_log.get('train/logits_real')),
                            "logits_fake": _to_csv_scalar(d_log.get('train/logits_fake')),
                            "r1_penalty": _to_csv_scalar(d_log.get('train/r1_penalty')),
                            "r1_effective_weight": _to_csv_scalar(d_log.get('train/r1_effective_weight')),
                            "nll_grads_norm": "",
                            "g_grads_norm": "",
                            "psnr": "",
                            "lpips": "",
                            "psnr_ema": "",
                            "lpips_ema": "",
                        })
                        csv_file.flush()
                        last_disc_csv_logged_step = current_step
                    except Exception as e:
                        logger.error(f"Failed to write discriminator metrics to CSV: {e}")

            update_bar(bar)
            current_step += 1

            # valid model
            def valid_model(model, name=""):
                nonlocal csv_warned_ema_legacy
                set_eval(modules_to_train)
                discriminator = disc.module.discriminator
                was_disc_training = discriminator.training
                discriminator.eval()
                try:
                    (
                        psnr_list,
                        lpips_list,
                        orig_images,
                        recon_images,
                        patch_records,
                        logged_sample_indices,
                    ) = valid(
                        global_rank, rank, model, discriminator, val_dataloader, precision, args,
                        lpips_model=val_lpips_model
                    )
                    valid_psnr, valid_lpips, valid_orig_images, valid_recon_images, valid_patch_records = gather_valid_result(
                        psnr_list, lpips_list, orig_images, recon_images, patch_records, rank, dist.get_world_size()
                    )
                finally:
                    set_train(modules_to_train)
                    discriminator.train(was_disc_training)

                if global_rank == 0:
                    is_ema = name == "ema"
                    name = "_" + name if name != "" else name
                    shared_sample_indices = [int(i) for i in logged_sample_indices[:args.eval_num_image_log]]

                    patch_score_dir = ckpt_dir / "val_patch_scores" / f"step_{current_step:08d}{name}"
                    recon_image_by_sample_idx = {}

                    # Create separate directories for original and reconstructed images
                    if len(valid_orig_images) > 0:
                        # Create directories
                        orig_dir = ckpt_dir / "val_images" / "original"
                        recon_dir = ckpt_dir / "val_images" / "reconstructed"
                        orig_dir.mkdir(exist_ok=True, parents=True)
                        recon_dir.mkdir(exist_ok=True, parents=True)

                        # Save individual images
                        max_shared = min(
                            args.eval_num_image_log,
                            len(valid_orig_images),
                            len(valid_recon_images),
                            len(shared_sample_indices),
                        )
                        for idx in range(max_shared):
                            sample_idx = shared_sample_indices[idx]
                            orig_img = valid_orig_images[idx]
                            recon_img = valid_recon_images[idx]
                            recon_uint8 = _tensor_image_to_uint8(recon_img)
                            if recon_uint8 is not None:
                                recon_image_by_sample_idx[sample_idx] = recon_uint8
                            save_image(
                                orig_img,
                                orig_dir / f"step_{current_step}_original{name}_{idx:03d}_sid{sample_idx}.png",
                            )
                            save_image(
                                recon_img,
                                recon_dir / f"step_{current_step}_recon{name}_{idx:03d}_sid{sample_idx}.png",
                            )

                        if use_wandb:
                            # Create grids for wandb logging
                            orig_grid = make_grid(valid_orig_images, nrow=4)
                            recon_grid = make_grid(valid_recon_images, nrow=4)
                            wandb.log(
                                {
                                    f"val{name}/original": wandb.Image(orig_grid),
                                    f"val{name}/reconstructed": wandb.Image(recon_grid),
                                },
                                step=current_step,
                            )

                    save_patch_scores(
                        valid_patch_records,
                        patch_score_dir,
                        val_dataset=val_dataset,
                        eval_resolution=args.eval_resolution,
                        vis_max_samples=args.eval_num_image_log,
                        selected_sample_indices=shared_sample_indices if len(shared_sample_indices) > 0 else None,
                        recon_image_by_sample_idx=recon_image_by_sample_idx if len(recon_image_by_sample_idx) > 0 else None,
                    )
                    log_patch_scores_to_wandb(valid_patch_records, current_step, f"val{name}", use_wandb=use_wandb)
                    if use_wandb:
                        wandb.log({f"val{name}/psnr": valid_psnr}, step=current_step)
                        wandb.log({f"val{name}/lpips": valid_lpips}, step=current_step)

                    # Log validation metrics to CSV
                    if csv_writer is not None:
                        try:
                            val_row = {
                                "step": current_step,
                                "generator_loss": "",
                                "discriminator_loss": "",
                                "rec_loss": "",
                                "perceptual_loss": "",
                                "kl_loss": "",
                                "wavelet_loss": "",
                                "logvar": "", "nll_loss": "",
                                "g_loss": "", "d_weight": "",
                                "logits_real": "", "logits_fake": "",
                                "r1_penalty": "", "r1_effective_weight": "",
                                "nll_grads_norm": "", "g_grads_norm": "",
                                "psnr": "",
                                "lpips": "",
                                "psnr_ema": "",
                                "lpips_ema": "",
                            }
                            should_write_val_row = True
                            if is_ema:
                                if csv_supports_ema_columns:
                                    val_row["psnr_ema"] = _to_csv_scalar(valid_psnr)
                                    val_row["lpips_ema"] = _to_csv_scalar(valid_lpips)
                                else:
                                    should_write_val_row = False
                                    if not csv_warned_ema_legacy:
                                        logger.warning(
                                            "Skipped EMA validation CSV row because resumed CSV "
                                            "does not have `psnr_ema/lpips_ema` columns."
                                        )
                                        csv_warned_ema_legacy = True
                            else:
                                val_row["psnr"] = _to_csv_scalar(valid_psnr)
                                val_row["lpips"] = _to_csv_scalar(valid_lpips)

                            if should_write_val_row:
                                csv_writer.writerow(val_row)
                                csv_file.flush()
                        except Exception as e:
                            logger.error(f"Failed to write validation metrics to CSV: {e}")

                    logger.info(f"{name} PatchGAN patch scores saved to `{patch_score_dir}` (summary.csv + patch_vis/*.png).")
                    logger.info(f"{name} Validation done.")

            if current_step % args.eval_steps == 0 or current_step == 1:
                if global_rank == 0:
                    logger.info("Starting validation...")
                valid_model(model)
                if args.ema:
                    ema.apply_shadow()
                    valid_model(model, "ema")
                    ema.restore()
                if global_rank == 0 and not args.disable_plot:
                    try:
                        plot_training_curves(
                            csv_path=ckpt_dir / csv_name,
                            output_path=ckpt_dir / "training_curves.png",
                            disc_start=args.disc_start
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update plot after validation: {e}")

            # save checkpoint
            if current_step % args.save_ckpt_step == 0 and global_rank == 0:
                file_path = save_checkpoint(
                    epoch,
                    current_step,
                    {
                        "gen_optimizer": gen_optimizer.state_dict(),
                        "disc_optimizer": disc_optimizer.state_dict(),
                    },
                    {
                        "gen_model": model.module.state_dict(),
                        "dics_model": disc.module.state_dict(),
                    },
                    scaler.state_dict(),
                    ddp_sampler.state_dict(),
                    ckpt_dir,
                    f"checkpoint-{current_step}.ckpt",
                    ema_state_dict=ema.shadow if args.ema else None,
                    disc_refresh_state={
                        "last_disc_refresh_step": last_disc_refresh_step,
                        "disc_loss_buffer": list(disc_loss_buffer),
                    } if args.disc_auto_refresh else None,
                )
                logger.info(f"Checkpoint has been saved to `{file_path}`.")

            if args.max_steps is not None and current_step >= args.max_steps:
                stop_training = True
                if global_rank == 0:
                    logger.info(f"Reached max_steps={args.max_steps}, stopping training.")
                break

        if stop_training:
            break

    # end training
    # Training completed - cleanup and final plot
    if global_rank == 0:
        # Close CSV file
        if csv_file is not None:
            try:
                csv_file.close()
                logger.info("CSV file closed.")
            except Exception as e:
                logger.error(f"Failed to close CSV: {e}")

        # Generate final plot
        if not args.disable_plot:
            try:
                logger.info("Generating final training curves plot...")
                plot_training_curves(
                    csv_path=ckpt_dir / csv_name,
                    output_path=ckpt_dir / "training_curves_final.png",
                    disc_start=args.disc_start
                )
            except Exception as e:
                logger.error(f"Failed to generate final plot: {e}")

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Distributed Image VAE Training")
    # Exp setting
    parser.add_argument(
        "--exp_name", type=str, default="test", help="experiment name"
    )
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    # Training setting
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="maximum training steps"
    )
    parser.add_argument("--save_ckpt_step", type=int, default=1000, help="save checkpoint every N steps")
    parser.add_argument("--ckpt_dir", type=str, default="./results/", help="checkpoint directory")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--log_steps", type=int, default=5, help="log every N steps")
    parser.add_argument("--freeze_encoder", action="store_true", help="freeze encoder during training")
    parser.add_argument("--clip_grad_norm", type=float, default=1e5, help="gradient clipping norm")

    # Data
    parser.add_argument("--image_path", type=str, default=None, help="path to training images or manifest file")
    parser.add_argument("--use_manifest", action="store_true", help="use manifest file instead of scanning directory")
    parser.add_argument("--resolution", type=int, default=256, help="image resolution")

    # Generator model
    parser.add_argument("--ignore_mismatched_sizes", action="store_true", help="ignore mismatched model sizes")
    parser.add_argument("--find_unused_parameters", action="store_true", help="find unused parameters in DDP")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default=None, help="path to pretrained model"
    )
    parser.add_argument("--model_name", type=str, default=None, help="model name to load from registry")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="path to checkpoint to resume from")
    parser.add_argument("--not_resume_training_process", action="store_true", help="don't resume training process")
    parser.add_argument("--model_config", type=str, default=None, help="path to model config JSON file")
    parser.add_argument(
        "--mix_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="precision for training",
    )
    parser.add_argument("--wavelet_loss", action="store_true", help="use wavelet loss")
    parser.add_argument("--not_resume_discriminator", action="store_true", help="don't resume discriminator")
    parser.add_argument("--not_resume_optimizer", action="store_true", help="don't resume optimizer")
    parser.add_argument("--wavelet_weight", type=float, default=0.1, help="weight for wavelet loss")

    # Discriminator Model
    parser.add_argument(
        "--disc_cls",
        type=str,
        default="causalimagevae.model.losses.LPIPSWithDiscriminator",
        help="discriminator class path",
    )
    parser.add_argument("--disc_start", type=int, default=80000, help="step to start discriminator training")
    parser.add_argument("--disc_weight", type=float, default=0.5, help="discriminator loss weight")
    parser.add_argument("--adaptive_weight_clamp", type=float, default=1e5, help="clamp upper bound for adaptive d_weight (default 1e5, lower to reduce disc influence)")
    parser.add_argument("--r1_weight", type=float, default=0.0, help="R1 gradient penalty weight for discriminator (0=disabled)")
    parser.add_argument("--r1_start", type=int, default=0, help="Step to start R1 penalty (0=from beginning)")
    parser.add_argument("--r1_warmup_steps", type=int, default=0, help="Linear warmup steps for R1 weight (0=no warmup, immediate full weight)")
    parser.add_argument("--disc_lr", type=float, default=None, help="discriminator learning rate (default: same as --lr, TTUR recommends 2-4x)")
    parser.add_argument("--refresh_discriminator", action="store_true", help="reinitialize discriminator weights when resuming from checkpoint")
    parser.add_argument("--disc_refresh_every", type=int, default=0, help="periodically refresh discriminator every N steps (0=disabled)")
    parser.add_argument("--disc_refresh_warmup", type=int, default=100, help="after refresh, train only discriminator for N steps")
    parser.add_argument("--disc_auto_refresh", action="store_true", help="auto-refresh discriminator when disc_loss stagnates at high value")
    parser.add_argument("--disc_stale_window", type=int, default=500, help="rolling window size (in disc steps) for staleness detection")
    parser.add_argument("--disc_stale_loss_threshold", type=float, default=0.8, help="mean disc_loss above this is considered 'high'")
    parser.add_argument("--disc_stale_std_threshold", type=float, default=0.02, help="std below this means loss is flat/stagnant")
    parser.add_argument("--disc_auto_refresh_cooldown", type=int, default=5000, help="minimum training steps between auto-refreshes")
    parser.add_argument("--kl_weight", type=float, default=1e-06, help="KL divergence weight")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="perceptual loss weight")
    parser.add_argument("--loss_type", type=str, default="l1", help="reconstruction loss type")
    parser.add_argument("--logvar_init", type=float, default=0.0, help="log variance initialization")
    parser.add_argument("--learn_logvar", action="store_true", help="learn the log variance (balances rec loss vs GAN loss automatically)")
    parser.add_argument("--logvar_lr", type=float, default=None, help="separate learning rate for logvar (default: same as --lr)")
    parser.add_argument("--csv_log_steps", type=int, default=50, help="log losses to CSV every N steps")
    parser.add_argument("--disable_plot", action="store_true", help="disable automatic plot generation")
    parser.add_argument("--disable_wandb", action="store_true", help="disable wandb logging")

    # Validation
    parser.add_argument("--eval_steps", type=int, default=1000, help="evaluate every N steps")
    parser.add_argument("--eval_image_path", type=str, default=None, help="path to validation images")
    parser.add_argument("--eval_resolution", type=int, default=256, help="validation image resolution")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="validation batch size")
    parser.add_argument("--eval_subset_size", type=int, default=30, help="number of validation samples (<=0 means full set)")
    parser.add_argument("--eval_num_image_log", type=int, default=20, help="number of images to log")
    parser.add_argument("--eval_lpips", action="store_true", help="compute LPIPS during validation")

    # Dataset
    parser.add_argument("--dataset_num_worker", type=int, default=4, help="number of data loading workers")

    # EMA
    parser.add_argument("--ema", action="store_true", help="use EMA")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")

    args = parser.parse_args()
    if args.csv_log_steps <= 0:
        raise ValueError(f"`--csv_log_steps` must be > 0, got {args.csv_log_steps}")

    set_random_seed(args.seed)
    train(args)

if __name__ == "__main__":
    main()

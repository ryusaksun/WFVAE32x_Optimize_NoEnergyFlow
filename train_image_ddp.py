import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, DistributedSampler, Subset
import argparse
import logging
import tqdm
from itertools import chain
from contextlib import nullcontext
import random
import numpy as np
from pathlib import Path
from causalimagevae.model import *
from causalimagevae.model.ema_model import EMA
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
    torch.save(data, filepath)
    return filepath

def valid(global_rank, rank, model, val_dataloader, precision, args, lpips_model=None):

    bar = None
    if global_rank == 0:
        bar = tqdm.tqdm(total=len(val_dataloader), desc="Validation...")

    psnr_records = []
    lpips_records = []
    image_records = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs = batch["image"].to(rank)
            sample_indices = batch.get("index")
            if sample_indices is None:
                start_idx = batch_idx * inputs.shape[0]
                dataset_size = len(val_dataloader.dataset)
                sample_indices = [
                    global_rank * dataset_size + start_idx + i
                    for i in range(inputs.shape[0])
                ]
            elif torch.is_tensor(sample_indices):
                sample_indices = sample_indices.tolist()
            else:
                sample_indices = [int(i) for i in sample_indices]

            with torch.amp.autocast("cuda", dtype=precision):
                output = model(inputs, sample_posterior=False)
                image_recon = output.sample

            # Only keep a small, globally stable subset of images for logging.
            for i in range(len(image_recon)):
                sample_idx = int(sample_indices[i])
                if 0 <= sample_idx < args.eval_num_image_log:
                    img_orig = (
                        ((inputs[i] + 1.0) / 2.0)
                        .clamp(0, 1)
                        .mul(255)
                        .round()
                        .to(torch.uint8)
                        .cpu()
                    )
                    img_recon = (
                        ((image_recon[i] + 1.0) / 2.0)
                        .clamp(0, 1)
                        .mul(255)
                        .round()
                        .to(torch.uint8)
                        .cpu()
                    )
                    image_records.append((sample_idx, img_orig, img_recon))

            # Calculate PSNR (data range [-1, 1], so MAX=2)
            mse = torch.mean(torch.square(inputs - image_recon), dim=(1, 2, 3))
            psnr_values = (
                20 * torch.log10(2.0 / torch.sqrt(mse.clamp(min=1e-10)))
            ).detach().cpu().tolist()
            psnr_records.extend(
                (int(sample_idx), float(psnr_value))
                for sample_idx, psnr_value in zip(sample_indices, psnr_values)
            )

            # Calculate LPIPS
            if args.eval_lpips:
                lpips_scores = (
                    lpips_model.forward(inputs, image_recon)
                    .detach()
                    .reshape(inputs.shape[0], -1)
                    .mean(dim=1)
                    .cpu()
                    .tolist()
                )
                lpips_records.extend(
                    (int(sample_idx), float(lpips_score))
                    for sample_idx, lpips_score in zip(sample_indices, lpips_scores)
                )

            if global_rank == 0:
                bar.update()

    return psnr_records, lpips_records, image_records

def gather_valid_result(psnr_records, lpips_records, image_records, global_rank, world_size, num_image_log):
    gathered_psnr_records = [None for _ in range(world_size)]
    gathered_lpips_records = [None for _ in range(world_size)]
    gathered_image_records = [None for _ in range(world_size)] if global_rank == 0 else None

    dist.all_gather_object(gathered_psnr_records, psnr_records)
    dist.all_gather_object(gathered_lpips_records, lpips_records)
    # Skip gather_object for image_records to avoid NCCL OOM on large models.
    # Only use rank 0's local image_records instead.

    all_psnr_records = list(chain(*gathered_psnr_records))
    all_lpips_records = list(chain(*gathered_lpips_records))

    seen_psnr = set()
    unique_psnr = []
    for idx, psnr_value in all_psnr_records:
        if idx not in seen_psnr:
            seen_psnr.add(idx)
            unique_psnr.append(psnr_value)

    seen_lpips = set()
    unique_lpips = []
    for idx, lpips_value in all_lpips_records:
        if idx not in seen_lpips:
            seen_lpips.add(idx)
            unique_lpips.append(lpips_value)

    unique_orig, unique_recon, unique_indices = [], [], []
    if global_rank == 0:
        seen = set()
        for idx, orig, recon in sorted(image_records, key=lambda record: record[0]):
            if idx in seen:
                continue
            seen.add(idx)
            unique_indices.append(idx)
            unique_orig.append(orig.float().div(255.0))
            unique_recon.append(recon.float().div(255.0))
            if len(unique_indices) >= num_image_log:
                break

    return (
        float(np.mean(unique_psnr)) if len(unique_psnr) > 0 else float("nan"),
        float(np.mean(unique_lpips)) if len(unique_lpips) > 0 else float("nan"),
        unique_orig,
        unique_recon,
        unique_indices,
    )

def _to_csv_scalar(value):
    if value is None or value == "":
        return ""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return ""
        return float(value.detach().float().mean().item())
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

    # Define metrics to plot
    metrics = [
        ('generator_loss', 'Generator Loss', 'tab:blue'),
        ('discriminator_loss', 'Discriminator Loss', 'tab:orange'),
        ('rec_loss', 'Reconstruction Loss', 'tab:green'),
        ('perceptual_loss', 'Perceptual Loss', 'tab:brown'),
        ('kl_loss', 'KL Loss', 'tab:red'),
        ('wavelet_loss', 'Wavelet Loss', 'tab:purple'),
        ('nll_loss', 'NLL Loss', 'slateblue'),
        ('g_loss', 'GAN Loss (g_loss)', 'tab:cyan'),
        ('d_weight', 'Adaptive Weight (d_weight)', 'darkgoldenrod'),
        ('logits_real', 'Logits Real', 'forestgreen'),
        ('logits_fake', 'Logits Fake', 'tomato'),
        ('nll_grads_norm', 'NLL Grad Norm', 'mediumpurple'),
        ('g_grads_norm', 'GAN Grad Norm', 'coral'),
        ('psnr', 'Validation PSNR', 'deepskyblue'),
        ('lpips', 'Validation LPIPS', 'tab:olive'),
        ('psnr_ema', 'Validation PSNR (EMA)', 'tab:pink'),
        ('lpips_ema', 'Validation LPIPS (EMA)', 'tab:gray'),
        ('active_channels', 'Active Channels (KL>0.01)', 'teal'),
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
    checkpoint_subdir = ckpt_dir / "checkpoints"
    if global_rank == 0:
        try:
            ckpt_dir.mkdir(exist_ok=False, parents=True)
        except FileExistsError:
            logger.warning(f"`{ckpt_dir}` exists!")
        checkpoint_subdir.mkdir(exist_ok=True)
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
        learn_logvar=args.learn_logvar,
        disc_type=args.disc_type,
        num_D=args.num_D,
        n_layers_D=args.n_layers_D,
        feat_match_weight=args.feat_match_weight,
        disc_norm=args.disc_norm,
    )
    if global_rank == 0:
        disc_params = sum(p.numel() for p in disc.discriminator.parameters())
        logger.info(
            f"discriminator: type={args.disc_type} norm={args.disc_norm}"
            + (f" num_D={args.num_D} n_layers_D={args.n_layers_D} "
               f"feat_match_weight={args.feat_match_weight}"
               if args.disc_type == "multiscale" else "")
            + f" | params={disc_params/1e6:.2f} M"
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
    aux_gen_params = []
    if disc.module.logvar.requires_grad:
        logvar_lr = args.logvar_lr if args.logvar_lr is not None else args.lr
        aux_gen_params = [disc.module.logvar]
        gen_param_groups = [
            {"params": parameters_to_train, "lr": args.lr},
            {"params": aux_gen_params, "lr": logvar_lr, "weight_decay": 0.0},
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
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu", weights_only=False)
        model.module.load_state_dict(checkpoint["state_dict"]["gen_model"], strict=False)

        # resume optimizer
        gen_optimizer.load_state_dict(checkpoint["optimizer_state"]["gen_optimizer"])
        if aux_gen_params and len(gen_optimizer.param_groups) >= 2:
            gen_optimizer.param_groups[0]["lr"] = args.lr
            gen_optimizer.param_groups[1]["lr"] = logvar_lr
            logger.info(
                f"Overriding resumed generator lr to {args.lr} "
                f"and logvar lr to {logvar_lr}"
            )
        else:
            for pg in gen_optimizer.param_groups:
                pg["lr"] = args.lr
            logger.info(f"Overriding resumed generator lr to {args.lr}")

        # resume discriminator
        disc.module.load_state_dict(checkpoint["state_dict"]["dics_model"])
        disc_optimizer.load_state_dict(checkpoint["optimizer_state"]["disc_optimizer"])
        # Override disc lr if --disc_lr is explicitly set (TTUR), since checkpoint saves the old lr
        if args.disc_lr is not None:
            for pg in disc_optimizer.param_groups:
                pg['lr'] = disc_lr
            logger.info(f"Overriding resumed discriminator lr to {disc_lr} (TTUR)")

        # M3: restore scaler state on resume, but only if scaler is enabled
        # and checkpoint has a non-empty scaler state (empty when saved with bf16/fp32)
        if "scaler_state" in checkpoint and scaler.is_enabled() and checkpoint["scaler_state"]:
            scaler.load_state_dict(checkpoint["scaler_state"])

        # resume data sampler and training progress
        ddp_sampler.load_state_dict(checkpoint["sampler_state"])
        start_epoch = checkpoint["sampler_state"]["epoch"]
        current_step = checkpoint["current_step"]
        logger.info(
            f"Checkpoint loaded from {args.resume_from_checkpoint}, starting from epoch {start_epoch} step {current_step}"
        )

    if args.ema:
        logger.warning(f"Start with EMA. EMA decay = {args.ema_decay}.")
        # Wrap model.module (not the DDP wrapper) so EMA shadow keys match
        # `gen_model: model.module.state_dict()` and are usable by inference
        # scripts that load the checkpoint without DDP.
        ema = EMA(model.module, args.ema_decay)
        ema.register()
        # Restore EMA shadow weights from checkpoint
        if args.resume_from_checkpoint and "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"]:
            loaded_shadow = checkpoint["ema_state_dict"]
            # Backward compat: old checkpoints (pre-fix) wrapped EMA around the
            # DDP model, so shadow keys had "module." prefix. Strip it so the
            # keys match model.module.named_parameters().
            loaded_shadow = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in loaded_shadow.items()
            }
            ema.shadow = loaded_shadow
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
            "fm_loss",
            "nll_loss",
            "g_loss", "d_weight",
            "logits_real", "logits_fake",
            "nll_grads_norm", "g_grads_norm",
            "psnr", "lpips", "psnr_ema", "lpips_ema",
            "active_channels",
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

    # Setup signal handlers for graceful interruption (flag-based, safe for NCCL)
    _interrupted = False

    def signal_handler(signum, frame):
        """Set flag for main loop to handle; avoid calling NCCL/sys.exit in handler."""
        nonlocal _interrupted
        _interrupted = True
        if global_rank == 0:
            logger.warning(f"\nReceived signal {signum}. Will exit at next safe point...")

    # Register signal handlers on all ranks
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
        val_lpips_model = lpips.LPIPS(net="alex", spatial=False)
        val_lpips_model.to(rank)
        val_lpips_model = DDP(val_lpips_model, device_ids=[rank])
        val_lpips_model.requires_grad_(False)
        val_lpips_model.eval()

    gradient_accumulation_steps = args.gradient_accumulation_steps

    # training bar
    bar_desc = "Epoch: {current_epoch}, Loss: {loss}"
    bar = None
    if global_rank == 0:
        steps_per_epoch = (
            len(dataloader) + gradient_accumulation_steps - 1
        ) // gradient_accumulation_steps
        max_steps = (
            args.epochs * steps_per_epoch if args.max_steps is None else args.max_steps
        )
        bar = tqdm.tqdm(total=max_steps, desc=bar_desc.format(current_epoch=0, loss=0))
        bar.update(current_step)
        logger.warning("Training Details: ")
        logger.warning(f" Max steps: {max_steps}")
        logger.warning(f" Dataset Samples: {len(dataset)}")
        effective_batch = args.batch_size * int(os.environ['WORLD_SIZE']) * gradient_accumulation_steps
        logger.warning(
            f" Total Batch Size: {args.batch_size} * {os.environ['WORLD_SIZE']} * {gradient_accumulation_steps} (accum) = {effective_batch}"
        )
    dist.barrier()

    num_epochs = args.epochs
    stop_training = False
    last_gen_csv_logged_step = -args.csv_log_steps
    last_disc_csv_logged_step = -args.csv_log_steps
    micro_step = 0
    step_gen = False
    step_dis = False
    g_loss = None
    d_loss = None
    g_log = {}
    d_log = {}
    posterior = None

    def update_bar(bar):
        if global_rank == 0:
            bar.desc = bar_desc.format(current_epoch=epoch, loss=f"-")
            bar.update()

    def rescale_optimizer_grads(optimizer, accum_steps):
        if accum_steps <= 0 or accum_steps == gradient_accumulation_steps:
            return
        scale_factor = gradient_accumulation_steps / accum_steps
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    param.grad.mul_(scale_factor)

    def average_optimizer_grads(optimizer):
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(world_size)

    def average_param_grads(params):
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        for param in params:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)

    def get_checkpoint_sampler_state(epoch, batch_idx, is_epoch_end):
        if is_epoch_end:
            return {
                "epoch": epoch + 1,
                "seed": ddp_sampler.seed,
                "current_index": 0,
            }
        next_index = min((batch_idx + 1) * args.batch_size, len(ddp_sampler))
        return {
            "epoch": epoch,
            "seed": ddp_sampler.seed,
            "current_index": next_index,
        }

    def run_post_step(epoch, batch_idx, is_epoch_end=False):
        nonlocal stop_training, csv_warned_ema_legacy

        update_bar(bar)

        # valid model
        def valid_model(model, name=""):
            nonlocal csv_warned_ema_legacy
            set_eval(modules_to_train)
            try:
                (
                    psnr_records,
                    lpips_records,
                    image_records,
                ) = valid(
                    global_rank, rank, model, val_dataloader, precision, args,
                    lpips_model=val_lpips_model
                )
                valid_psnr, valid_lpips, valid_orig_images, valid_recon_images, logged_sample_indices = gather_valid_result(
                    psnr_records,
                    lpips_records,
                    image_records,
                    global_rank,
                    dist.get_world_size(),
                    args.eval_num_image_log,
                )
            finally:
                set_train(modules_to_train)

            if global_rank == 0:
                is_ema = name == "ema"
                name = "_" + name if name != "" else name

                # Create separate directories for original and reconstructed images
                if len(valid_orig_images) > 0:
                    # Create directories
                    orig_dir = ckpt_dir / "val_images" / "original"
                    recon_dir = ckpt_dir / "val_images" / "reconstructed"
                    orig_dir.mkdir(exist_ok=True, parents=True)
                    recon_dir.mkdir(exist_ok=True, parents=True)

                    # Save individual images
                    shared_sample_indices = [int(i) for i in logged_sample_indices[:args.eval_num_image_log]]
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
                        orig_grid = make_grid(valid_orig_images[:args.eval_num_image_log], nrow=4)
                        recon_grid = make_grid(valid_recon_images[:args.eval_num_image_log], nrow=4)
                        wandb.log(
                            {
                                f"val{name}/original": wandb.Image(orig_grid),
                                f"val{name}/reconstructed": wandb.Image(recon_grid),
                            },
                            step=current_step,
                        )

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
                            "fm_loss": "",
                            "nll_loss": "",
                            "g_loss": "", "d_weight": "",
                            "logits_real": "", "logits_fake": "",
                            "nll_grads_norm": "", "g_grads_norm": "",
                            "psnr": "",
                            "lpips": "",
                            "psnr_ema": "",
                            "lpips_ema": "",
                            "active_channels": "",
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

                logger.info(f"{name} Validation done.")

        if current_step % args.eval_steps == 0 or current_step == 1:
            if global_rank == 0:
                logger.info("Starting validation...")
            valid_model(model)
            if args.ema:
                ema.apply_shadow()
                try:
                    valid_model(model, "ema")
                finally:
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
            sampler_state = get_checkpoint_sampler_state(
                epoch=epoch,
                batch_idx=batch_idx,
                is_epoch_end=is_epoch_end,
            )
            file_path = save_checkpoint(
                sampler_state["epoch"],
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
                sampler_state,
                checkpoint_subdir,
                f"checkpoint-{current_step}.ckpt",
                ema_state_dict=ema.shadow if args.ema else None,
            )
            logger.info(f"Checkpoint has been saved to `{file_path}`.")

        if args.max_steps is not None and current_step >= args.max_steps:
            stop_training = True
            if global_rank == 0:
                logger.info(f"Reached max_steps={args.max_steps}, stopping training.")

    def finish_accumulation(epoch, accum_steps, batch_idx, is_epoch_end=False):
        nonlocal current_step, last_gen_csv_logged_step, last_disc_csv_logged_step

        if accum_steps <= 0:
            return

        current_step += 1

        # --- Optimizer step (after full or partial accumulation) ---
        if step_gen:
            scaler.unscale_(gen_optimizer)
            rescale_optimizer_grads(gen_optimizer, accum_steps)
            # M4: clip all params in gen_optimizer (model + logvar)
            all_gen_params = [p for pg in gen_optimizer.param_groups for p in pg["params"]]
            torch.nn.utils.clip_grad_norm_(all_gen_params, args.clip_grad_norm)
            _scale_before = scaler.get_scale()
            scaler.step(gen_optimizer)
            scaler.update()

            # update ema only if optimizer actually stepped (no inf/nan grads)
            if args.ema and scaler.get_scale() >= _scale_before:
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
                    {"train/latents_std": posterior.mean.detach().std().item()}, step=current_step
                )
                wandb.log(
                    {"train/active_channels": _to_csv_scalar(g_log.get('train/active_channels', 0))}, step=current_step
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
                        "fm_loss": _to_csv_scalar(g_log.get('train/fm_loss')),
                        "nll_loss": _to_csv_scalar(g_log.get('train/nll_loss')),
                        "g_loss": _to_csv_scalar(g_log.get('train/g_loss')),
                        "d_weight": _to_csv_scalar(g_log.get('train/d_weight')),
                        "logits_real": "",
                        "logits_fake": "",
                        "nll_grads_norm": _to_csv_scalar(g_log.get('train/nll_grads_norm')),
                        "g_grads_norm": _to_csv_scalar(g_log.get('train/g_grads_norm')),
                        "psnr": "",
                        "lpips": "",
                        "psnr_ema": "",
                        "lpips_ema": "",
                        "active_channels": _to_csv_scalar(g_log.get('train/active_channels')),
                    })
                    csv_file.flush()
                    last_gen_csv_logged_step = current_step
                except Exception as e:
                    logger.error(f"Failed to write generator metrics to CSV: {e}")

        if step_dis:
            scaler.unscale_(disc_optimizer)
            rescale_optimizer_grads(disc_optimizer, accum_steps)
            torch.nn.utils.clip_grad_norm_(disc.module.discriminator.parameters(), args.clip_grad_norm)
            scaler.step(disc_optimizer)
            scaler.update()
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
                        "fm_loss": "",
                        "nll_loss": "",
                        "g_loss": "",
                        "d_weight": "",
                        "logits_real": _to_csv_scalar(d_log.get('train/logits_real')),
                        "logits_fake": _to_csv_scalar(d_log.get('train/logits_fake')),
                        "nll_grads_norm": "",
                        "g_grads_norm": "",
                        "psnr": "",
                        "lpips": "",
                        "psnr_ema": "",
                        "lpips_ema": "",
                        "active_channels": "",
                    })
                    csv_file.flush()
                    last_disc_csv_logged_step = current_step
                except Exception as e:
                    logger.error(f"Failed to write discriminator metrics to CSV: {e}")

        run_post_step(epoch, batch_idx=batch_idx, is_epoch_end=is_epoch_end)

    # training Loop
    try:  # M2: try/finally guarantees CSV close on any exception
     for epoch in range(start_epoch, num_epochs):
        set_train(modules_to_train)
        ddp_sampler.set_epoch(epoch)  # Shuffle data at every epoch

        for batch_idx, batch in enumerate(dataloader):
            # M1: check interrupt flag at safe point (between iterations, outside NCCL ops)
            if _interrupted:
                stop_training = True
                break

            if args.max_steps is not None and current_step >= args.max_steps:
                stop_training = True
                break

            inputs = batch["image"].to(rank)

            # --- At start of accumulation cycle: select step type and zero grad ---
            if micro_step == 0:
                if (
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

                if step_gen:
                    gen_optimizer.zero_grad()
                if step_dis:
                    disc_optimizer.zero_grad()

            micro_step += 1
            is_last_micro = (micro_step == gradient_accumulation_steps)

            # DDP gradient sync only on last micro-step
            sync_ctx = model.no_sync() if (step_gen and not is_last_micro) else nullcontext()
            disc_sync_ctx = disc.no_sync() if (step_dis and not is_last_micro) else nullcontext()

            with sync_ctx, disc_sync_ctx:
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
                    scaler.scale(g_loss / gradient_accumulation_steps).backward()

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
                    scaler.scale(d_loss / gradient_accumulation_steps).backward()

            # Skip to next micro-batch if accumulation not complete
            if not is_last_micro:
                continue
            if step_gen and aux_gen_params:
                average_param_grads(aux_gen_params)
            finish_accumulation(
                epoch,
                micro_step,
                batch_idx=batch_idx,
                is_epoch_end=(batch_idx + 1 == len(dataloader)),
            )
            micro_step = 0
            if stop_training:
                break

        if micro_step > 0:
            if global_rank == 0:
                logger.info(
                    "Flushing partial gradient accumulation at epoch end: "
                    f"{micro_step}/{gradient_accumulation_steps} micro-batches."
                )
            if step_gen:
                average_optimizer_grads(gen_optimizer)
            if step_dis:
                average_optimizer_grads(disc_optimizer)
            finish_accumulation(
                epoch,
                micro_step,
                batch_idx=batch_idx,
                is_epoch_end=True,
            )
            micro_step = 0

        if stop_training:
            break

    finally:  # M2: guarantee CSV close on any exception / interrupt / normal exit
        if global_rank == 0 and csv_file is not None:
            try:
                csv_file.flush()
                csv_file.close()
                logger.info("CSV file closed.")
            except Exception as e:
                logger.error(f"Failed to close CSV: {e}")

    # end training — cleanup and final plot
    try:
        if global_rank == 0:
            # Generate final plot (suffix depends on whether interrupted)
            if not args.disable_plot:
                try:
                    if _interrupted:
                        plot_name = f"training_curves_interrupted_step{current_step}.png"
                    else:
                        plot_name = "training_curves_final.png"
                    logger.info(f"Generating training curves plot: {plot_name}")
                    plot_training_curves(
                        csv_path=ckpt_dir / csv_name,
                        output_path=ckpt_dir / plot_name,
                        disc_start=args.disc_start
                    )
                except Exception as e:
                    logger.error(f"Failed to generate final plot: {e}")
    finally:
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="number of micro-batches to accumulate before optimizer step")

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
    parser.add_argument("--model_config", type=str, default=None, help="path to model config JSON file")
    parser.add_argument(
        "--mix_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="precision for training",
    )
    parser.add_argument("--wavelet_loss", action="store_true", help="use wavelet loss")
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
    parser.add_argument("--disc_lr", type=float, default=None, help="discriminator learning rate (default: same as --lr, TTUR recommends 2-4x)")
    parser.add_argument("--disc_type", type=str, default="single", choices=["single", "multiscale"], help="discriminator variant: 'single' = original PatchGAN (default), 'multiscale' = pix2pixHD-style N-scale disc + feature matching")
    parser.add_argument("--num_D", type=int, default=3, help="number of discriminator scales (multiscale only)")
    parser.add_argument("--n_layers_D", type=int, default=3, help="number of conv layers per discriminator (multiscale only)")
    parser.add_argument("--feat_match_weight", type=float, default=10.0, help="pix2pixHD feature-matching loss weight (multiscale only; 0 disables)")
    parser.add_argument("--disc_norm", type=str, default="bn", choices=["bn", "sn", "in", "none"], help="discriminator normalization: bn (default, BatchNorm2d), sn (spectral_norm + Identity, Miyato et al. 2018), in (InstanceNorm2d), none (Identity, pure ablation)")
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
    if args.gradient_accumulation_steps <= 0:
        raise ValueError(
            "`--gradient_accumulation_steps` must be > 0, "
            f"got {args.gradient_accumulation_steps}"
        )

    set_random_seed(args.seed)
    train(args)

if __name__ == "__main__":
    main()

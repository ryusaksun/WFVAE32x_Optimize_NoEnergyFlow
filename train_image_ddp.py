import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, DistributedSampler, Subset
import argparse
import logging
import tqdm
from itertools import chain
import wandb
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
    import lpips
except:
    raise Exception("Need lpips to validate.")

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
    ema_state_dict={},
):
    filepath = checkpoint_dir / Path(filename)
    torch.save(
        {
            "epoch": epoch,
            "current_step": current_step,
            "optimizer_state": optimizer_state,
            "state_dict": state_dict,
            "ema_state_dict": ema_state_dict,
            "scaler_state": scaler_state,
            "sampler_state": sampler_state,
        },
        filepath,
    )
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

def save_patch_scores(patch_records, output_dir):
    if len(patch_records) == 0:
        return

    output_dir.mkdir(exist_ok=True, parents=True)
    ordered_records = sorted(patch_records, key=lambda x: int(x["sample_idx"]))

    real_logits_array = np.stack(
        [np.asarray(record["real_logits"], dtype=np.float32) for record in ordered_records],
        axis=0,
    )
    recon_logits_array = np.stack(
        [np.asarray(record["recon_logits"], dtype=np.float32) for record in ordered_records],
        axis=0,
    )
    real_sigmoid_array = np.stack(
        [np.asarray(record["real_sigmoid"], dtype=np.float32) for record in ordered_records],
        axis=0,
    )
    recon_sigmoid_array = np.stack(
        [np.asarray(record["recon_sigmoid"], dtype=np.float32) for record in ordered_records],
        axis=0,
    )

    np.save(output_dir / "real_logits.npy", real_logits_array)
    np.save(output_dir / "recon_logits.npy", recon_logits_array)
    np.save(output_dir / "real_sigmoid.npy", real_sigmoid_array)
    np.save(output_dir / "recon_sigmoid.npy", recon_sigmoid_array)

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

def log_patch_scores_to_wandb(patch_records, step, prefix):
    if len(patch_records) == 0:
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

def valid(global_rank, rank, model, discriminator, val_dataloader, precision, args):
    if args.eval_lpips:
        lpips_model = lpips.LPIPS(net="alex", spatial=True)
        lpips_model.to(rank)
        lpips_model = DDP(lpips_model, device_ids=[rank])
        lpips_model.requires_grad_(False)
        lpips_model.eval()

    bar = None
    if global_rank == 0:
        bar = tqdm.tqdm(total=len(val_dataloader), desc="Validation...")

    psnr_list = []
    lpips_list = []
    orig_images = []
    recon_images = []
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
                    num_image_log -= 1

            # Calculate PSNR
            mse = torch.mean(torch.square(inputs - image_recon), dim=(1, 2, 3))
            psnr = 20 * torch.log10(1 / torch.sqrt(mse))
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
    return psnr_list, lpips_list, orig_images, recon_images, patch_records

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

def plot_training_curves(csv_path, output_path, disc_start=None, smoothing_window=50):
    """
    Plots training curves from CSV log file with 3x3 multi-subplot layout.

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
        ('kl_loss', 'KL Loss', 'tab:red'),
        ('wavelet_loss', 'Wavelet Loss', 'tab:purple'),
        ('psnr', 'Validation PSNR', 'tab:cyan'),
        ('lpips', 'Validation LPIPS', 'tab:olive'),
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
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Get data and remove NaN values
        data = df[['step', metric_key]].dropna()

        if len(data) == 0:
            ax.text(0.5, 0.5, f'{title}\n(No data)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        steps = data['step'].values
        values = data[metric_key].values

        # Plot raw data with transparency
        ax.plot(steps, values, alpha=0.3, color=color, linewidth=0.5, label='Raw')

        # Apply smoothing if we have enough data points
        if len(values) > smoothing_window:
            try:
                smoothed = uniform_filter1d(values, size=smoothing_window, mode='nearest')
                ax.plot(steps, smoothed, color=color, linewidth=2, label=f'Smoothed (w={smoothing_window})')
                ax.legend(loc='best', fontsize=8)
            except Exception as e:
                print(f"Warning: Could not smooth {metric_key}: {e}")

        # Draw discriminator start line if applicable
        if disc_start is not None and disc_start > 0:
            ax.axvline(x=disc_start, color='red', linestyle='--',
                      linewidth=1, alpha=0.5, label=f'Disc Start ({disc_start})')

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

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

    if global_rank == 0:
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
        wavelet_weight=args.wavelet_weight
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

    gen_optimizer = torch.optim.AdamW(parameters_to_train, lr=args.lr, weight_decay=args.weight_decay)
    disc_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, disc.module.discriminator.parameters()), lr=args.lr, weight_decay=args.weight_decay
    )

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda')
    precision = torch.bfloat16
    if args.mix_precision == "fp16":
        precision = torch.float16
    elif args.mix_precision == "fp32":
        precision = torch.float32

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
        if not args.not_resume_discriminator:
            disc.module.load_state_dict(checkpoint["state_dict"]["dics_model"])
            disc_optimizer.load_state_dict(checkpoint["optimizer_state"]["disc_optimizer"])
            scaler.load_state_dict(checkpoint["scaler_state"])

        # resume data sampler
        ddp_sampler.load_state_dict(checkpoint["sampler_state"])

        start_epoch = checkpoint["sampler_state"]["epoch"]
        current_step = checkpoint["current_step"]
        logger.info(
            f"Checkpoint loaded from {args.resume_from_checkpoint}, starting from epoch {start_epoch} step {current_step}"
        )

    if args.ema:
        logger.warning(f"Start with EMA. EMA decay = {args.ema_decay}.")
        ema = EMA(model, args.ema_decay)
        ema.register()

    logger.info("Prepared!")

    # Initialize CSV logger
    csv_file = None
    csv_writer = None
    if global_rank == 0:
        csv_path = ckpt_dir / "training_losses.csv"
        fieldnames = [
            "step", "generator_loss", "discriminator_loss",
            "rec_loss", "kl_loss", "wavelet_loss", "psnr", "lpips"
        ]

        # Check if resuming from checkpoint
        if args.resume_from_checkpoint and csv_path.exists():
            logger.info(f"Resuming CSV logging to {csv_path}")
            csv_file = open(csv_path, "a", newline="")
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        else:
            logger.info(f"Starting new CSV log at {csv_path}")
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

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
                    csv_path = ckpt_dir / "training_losses.csv"
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

    # Register signal handlers (only on rank 0 to avoid race conditions)
    if global_rank == 0:
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

    def update_bar(bar):
        if global_rank == 0:
            bar.desc = bar_desc.format(current_epoch=epoch, loss=f"-")
            bar.update()

    # training Loop
    for epoch in range(num_epochs):
        set_train(modules_to_train)
        ddp_sampler.set_epoch(epoch)  # Shuffle data at every epoch

        for batch_idx, batch in enumerate(dataloader):
            if args.max_steps is not None and current_step >= args.max_steps:
                stop_training = True
                break

            inputs = batch["image"].to(rank)

            # select generator or discriminator
            if (
                current_step % 2 == 1
                and current_step >= disc.module.discriminator_iter_start
            ):
                set_modules_requires_grad(modules_to_train, False)
                step_gen = False
                step_dis = True
            else:
                set_modules_requires_grad(modules_to_train, True)
                step_gen = True
                step_dis = False

            assert (
                step_gen or step_dis
            ), "You should backward either Gen. or Dis. in a step."

            # forward
            with torch.amp.autocast('cuda', dtype=precision):
                outputs = model(inputs)
                recon = outputs.sample
                posterior = outputs.latent_dist
                wavelet_coeffs = None
                if outputs.extra_output is not None and args.wavelet_loss:
                    wavelet_coeffs = outputs.extra_output

            # generator loss
            if step_gen:
                with torch.amp.autocast('cuda', dtype=precision):
                    g_loss, g_log = disc(
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
                scaler.step(gen_optimizer)
                scaler.update()

                # update ema
                if args.ema:
                    ema.update()

                # log to wandb
                if global_rank == 0 and current_step % args.log_steps == 0:
                    wandb.log(
                        {"train/generator_loss": g_loss.item()}, step=current_step
                    )
                    wandb.log(
                        {"train/rec_loss": g_log['train/rec_loss']}, step=current_step
                    )
                    wandb.log(
                        {"train/latents_std": posterior.sample().std().item()}, step=current_step
                    )

                # Log to CSV
                if current_step % args.csv_log_steps == 0 and csv_writer is not None:
                    try:
                        csv_writer.writerow({
                            "step": current_step,
                            "generator_loss": g_loss.item(),
                            "discriminator_loss": "",
                            "rec_loss": g_log.get('train/rec_loss', ""),
                            "kl_loss": g_log.get('train/kl_loss', ""),
                            "wavelet_loss": g_log.get('train/wl_loss', ""),
                            "psnr": "",
                            "lpips": ""
                        })
                        csv_file.flush()
                    except Exception as e:
                        logger.error(f"Failed to write CSV: {e}")

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
                scaler.step(disc_optimizer)
                scaler.update()
                if global_rank == 0 and current_step % args.log_steps == 0:
                    wandb.log(
                        {"train/discriminator_loss": d_loss.item()}, step=current_step
                    )

                    # Log to CSV
                    if current_step % args.csv_log_steps == 0 and csv_writer is not None:
                        try:
                            csv_writer.writerow({
                                "step": current_step,
                                "generator_loss": "",
                                "discriminator_loss": d_loss.item(),
                                "rec_loss": "",
                                "kl_loss": "",
                                "wavelet_loss": "",
                                "psnr": "",
                                "lpips": ""
                            })
                            csv_file.flush()
                        except Exception as e:
                            logger.error(f"Failed to write CSV: {e}")

            update_bar(bar)
            current_step += 1

            # valid model
            def valid_model(model, name=""):
                set_eval(modules_to_train)
                discriminator = disc.module.discriminator
                was_disc_training = discriminator.training
                discriminator.eval()
                try:
                    psnr_list, lpips_list, orig_images, recon_images, patch_records = valid(
                        global_rank, rank, model, discriminator, val_dataloader, precision, args
                    )
                    valid_psnr, valid_lpips, valid_orig_images, valid_recon_images, valid_patch_records = gather_valid_result(
                        psnr_list, lpips_list, orig_images, recon_images, patch_records, rank, dist.get_world_size()
                    )
                finally:
                    set_train(modules_to_train)
                    discriminator.train(was_disc_training)

                if global_rank == 0:
                    name = "_" + name if name != "" else name

                    patch_score_dir = ckpt_dir / "val_patch_scores" / f"step_{current_step:08d}{name}"
                    save_patch_scores(valid_patch_records, patch_score_dir)
                    log_patch_scores_to_wandb(valid_patch_records, current_step, f"val{name}")

                    # Create separate directories for original and reconstructed images
                    if len(valid_orig_images) > 0:
                        # Create directories
                        orig_dir = ckpt_dir / "val_images" / "original"
                        recon_dir = ckpt_dir / "val_images" / "reconstructed"
                        orig_dir.mkdir(exist_ok=True, parents=True)
                        recon_dir.mkdir(exist_ok=True, parents=True)

                        # Save individual images
                        for idx, (orig_img, recon_img) in enumerate(zip(valid_orig_images, valid_recon_images)):
                            save_image(orig_img, orig_dir / f"step_{current_step}_original{name}_{idx}.png")
                            save_image(recon_img, recon_dir / f"step_{current_step}_recon{name}_{idx}.png")

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
                    wandb.log({f"val{name}/psnr": valid_psnr}, step=current_step)
                    wandb.log({f"val{name}/lpips": valid_lpips}, step=current_step)

                    # Log validation metrics to CSV
                    if csv_writer is not None:
                        try:
                            csv_writer.writerow({
                                "step": current_step,
                                "generator_loss": "",
                                "discriminator_loss": "",
                                "rec_loss": "",
                                "kl_loss": "",
                                "wavelet_loss": "",
                                "psnr": valid_psnr,
                                "lpips": valid_lpips
                            })
                            csv_file.flush()
                        except Exception as e:
                            logger.error(f"Failed to write validation metrics to CSV: {e}")

                    logger.info(f"{name} PatchGAN patch scores saved to `{patch_score_dir}`.")
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
                            csv_path=ckpt_dir / "training_losses.csv",
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
                    ema_state_dict=ema.shadow if args.ema else {},
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
                    csv_path=ckpt_dir / "training_losses.csv",
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
    parser.add_argument("--load_disc_from_checkpoint", type=str, default=None, help="load discriminator from checkpoint")
    parser.add_argument(
        "--disc_cls",
        type=str,
        default="causalimagevae.model.losses.LPIPSWithDiscriminator",
        help="discriminator class path",
    )
    parser.add_argument("--disc_start", type=int, default=80000, help="step to start discriminator training")
    parser.add_argument("--disc_weight", type=float, default=0.5, help="discriminator loss weight")
    parser.add_argument("--kl_weight", type=float, default=1e-06, help="KL divergence weight")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="perceptual loss weight")
    parser.add_argument("--loss_type", type=str, default="l1", help="reconstruction loss type")
    parser.add_argument("--logvar_init", type=float, default=0.0, help="log variance initialization")
    parser.add_argument("--csv_log_steps", type=int, default=50, help="log losses to CSV every N steps")
    parser.add_argument("--disable_plot", action="store_true", help="disable automatic plot generation")

    # Validation
    parser.add_argument("--eval_steps", type=int, default=1000, help="evaluate every N steps")
    parser.add_argument("--eval_image_path", type=str, default=None, help="path to validation images")
    parser.add_argument("--eval_resolution", type=int, default=256, help="validation image resolution")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="validation batch size")
    parser.add_argument("--eval_subset_size", type=int, default=100, help="number of validation samples (<=0 means full set)")
    parser.add_argument("--eval_num_image_log", type=int, default=8, help="number of images to log")
    parser.add_argument("--eval_lpips", action="store_true", help="compute LPIPS during validation")

    # Dataset
    parser.add_argument("--dataset_num_worker", type=int, default=4, help="number of data loading workers")

    # EMA
    parser.add_argument("--ema", action="store_true", help="use EMA")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")

    args = parser.parse_args()

    set_random_seed(args.seed)
    train(args)

if __name__ == "__main__":
    main()

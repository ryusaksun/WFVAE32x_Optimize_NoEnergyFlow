# Training Loss Logging and Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CSV loss logging, automatic plot generation with signal handling, and update discriminator start parameter to 20000.

**Architecture:** Extend train_image_ddp.py with CSV writer, matplotlib plotting function, signal handlers, and periodic plot generation at checkpoints. No external dependencies beyond matplotlib/pandas/scipy.

**Tech Stack:** Python 3.10, PyTorch DDP, matplotlib, pandas, scipy

---

## Task 1: Update Discriminator Start Parameter

**Files:**
- Modify: `train_image_ddp.py:636`

**Step 1: Update default value**

Change line 636 from:
```python
parser.add_argument("--disc_start", type=int, default=5000, help="step to start discriminator training")
```

To:
```python
parser.add_argument("--disc_start", type=int, default=20000, help="step to start discriminator training")
```

**Step 2: Verify change**

Read `train_image_ddp.py:636` and confirm the default value is 20000.

**Step 3: Commit**

```bash
git add train_image_ddp.py
git commit -m "feat: increase discriminator start to 20000 steps

Allows generator more time to learn basic reconstruction before
adversarial training begins, reducing risk of mode collapse."
```

---

## Task 2: Add New Command Line Arguments

**Files:**
- Modify: `train_image_ddp.py:641-658` (after existing arguments)

**Step 1: Add csv_log_steps argument**

Add after line 641 (after `--logvar_init`):
```python
parser.add_argument("--csv_log_steps", type=int, default=50, help="log losses to CSV every N steps")
```

**Step 2: Add disable_plot argument**

Add after the csv_log_steps argument:
```python
parser.add_argument("--disable_plot", action="store_true", help="disable automatic plot generation")
```

**Step 3: Verify arguments**

Check that both arguments are added in the Validation section (before `# Validation` comment).

**Step 4: Commit**

```bash
git add train_image_ddp.py
git commit -m "feat: add csv_log_steps and disable_plot arguments

- csv_log_steps: configurable CSV logging frequency (default 50)
- disable_plot: option to skip automatic plot generation"
```

---

## Task 3: Add Plotting Function

**Files:**
- Modify: `train_image_ddp.py` (add function before `train()` function, around line 189)

**Step 1: Import required libraries**

Add to imports section (after line 25):
```python
import csv
import signal
import sys
```

**Step 2: Add plot_training_curves function**

Insert before the `train()` function definition (around line 189):
```python
def plot_training_curves(csv_path, output_path, disc_start=20000, logger=None):
    """Generate training curves plot from CSV file.

    Args:
        csv_path: Path to training_losses.csv
        output_path: Path to save plot PNG
        disc_start: Step when discriminator training starts (for vertical line marker)
        logger: Optional logger for messages
    """
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from scipy.ndimage import uniform_filter1d
    except ImportError as e:
        if logger:
            logger.warning(f"Cannot generate plot: {e}")
            logger.warning("Install with: pip install pandas matplotlib scipy")
        return

    if not csv_path.exists():
        if logger:
            logger.warning(f"CSV file not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        if logger:
            logger.warning(f"Failed to read CSV: {e}")
        return

    if len(df) < 2:
        if logger:
            logger.warning("Not enough data points to plot (need at least 2)")
        return

    # Define metrics to plot
    metrics = [
        ("generator_loss", "Generator Loss"),
        ("discriminator_loss", "Discriminator Loss"),
        ("rec_loss", "Reconstruction Loss"),
        ("kl_loss", "KL Divergence"),
        ("wavelet_loss", "Wavelet Loss"),
        ("psnr", "PSNR (dB)"),
        ("lpips", "LPIPS"),
    ]

    # Create 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]

        # Filter out missing values
        data = df[["step", metric]].dropna()

        if len(data) > 0:
            steps = data["step"].values
            values = data[metric].values

            # Plot smoothed + raw data
            if len(values) > 10:
                smoothed = uniform_filter1d(values, size=10, mode='nearest')
                ax.plot(steps, smoothed, linewidth=2, label="Smoothed", color='#2E86AB')
                ax.plot(steps, values, alpha=0.3, linewidth=1, label="Raw", color='#A23B72')
            else:
                ax.plot(steps, values, linewidth=2, color='#2E86AB')

            # Add discriminator start marker
            if disc_start > 0:
                ax.axvline(x=disc_start, color='red', linestyle='--',
                          linewidth=1.5, alpha=0.7, label=f'Disc Start ({disc_start})')

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel("Training Step", fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=8)
            ax.tick_params(labelsize=9)
        else:
            # No data for this metric
            ax.text(0.5, 0.5, f'No data for {title}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')

    # Hide unused subplots
    for idx in range(len(metrics), 9):
        axes[idx].axis('off')

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        if logger:
            logger.info(f"Plot saved to {output_path}")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save plot: {e}")
        plt.close()
```

**Step 3: Verify function**

Read the function back to ensure it's correctly added.

**Step 4: Commit**

```bash
git add train_image_ddp.py
git commit -m "feat: add plot_training_curves function

Generates 3x3 multi-subplot visualization of training losses:
- Supports 7 metrics (generator, discriminator, rec, kl, wavelet, psnr, lpips)
- Smoothed + raw data overlay
- Discriminator start marker (vertical line)
- High-resolution output (300 DPI)
- Graceful error handling"
```

---

## Task 4: Initialize CSV Writer in Training Function

**Files:**
- Modify: `train_image_ddp.py:378-403` (after logger setup, before training bar)

**Step 1: Add CSV initialization code**

After line 378 (`logger.info("Prepared!")`), before `dist.barrier()`, add:

```python
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
```

**Step 2: Verify code placement**

Ensure the code is added after "Prepared!" log but before the barrier.

**Step 3: Commit**

```bash
git add train_image_ddp.py
git commit -m "feat: initialize CSV writer for loss logging

- Creates training_losses.csv in checkpoint directory
- Supports resume mode (append to existing file)
- Only runs on rank 0 (avoids multi-process conflicts)"
```

---

## Task 5: Add Signal Handlers for Graceful Interruption

**Files:**
- Modify: `train_image_ddp.py` (after CSV initialization, before training bar)

**Step 1: Define signal handler function**

After the CSV initialization code (from Task 4), add:

```python
    # Setup signal handlers for graceful interruption
    def setup_signal_handlers():
        """Setup handlers to catch Ctrl+C and generate plot before exit"""
        def signal_handler(signum, frame):
            if global_rank == 0:
                logger.info(f"Received signal {signum}. Cleaning up...")

                # Close CSV file
                if csv_file is not None:
                    try:
                        csv_file.close()
                        logger.info("CSV file closed.")
                    except:
                        pass

                # Generate plot
                if not args.disable_plot:
                    try:
                        logger.info("Generating plot before exit...")
                        plot_training_curves(
                            csv_path=ckpt_dir / "training_losses.csv",
                            output_path=ckpt_dir / "training_curves_interrupted.png",
                            disc_start=args.disc_start,
                            logger=logger
                        )
                    except Exception as e:
                        logger.error(f"Failed to generate plot: {e}")

            # Cleanup distributed
            try:
                dist.destroy_process_group()
            except:
                pass

            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # kill command

    if global_rank == 0:
        setup_signal_handlers()
        logger.info("Signal handlers registered (Ctrl+C will save plot)")
```

**Step 2: Verify placement**

Ensure signal handlers are set up after CSV initialization.

**Step 3: Commit**

```bash
git add train_image_ddp.py
git commit -m "feat: add signal handlers for graceful interruption

Catches SIGINT (Ctrl+C) and SIGTERM (kill):
- Closes CSV file properly
- Generates plot with '_interrupted' suffix
- Cleans up distributed resources
- Provides graceful exit"
```

---

## Task 6: Add CSV Logging in Training Loop (Generator)

**Files:**
- Modify: `train_image_ddp.py:468-479` (in generator loss logging section)

**Step 1: Add CSV logging after WandB logs**

After line 478 (after the latents_std wandb.log), add:

```python

                    # Log to CSV
                    if current_step % args.csv_log_steps == 0:
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
```

**Step 2: Verify indentation**

Ensure the code is properly indented (inside the `if global_rank == 0 and current_step % args.log_steps == 0:` block).

**Step 3: Commit**

```bash
git add train_image_ddp.py
git commit -m "feat: add CSV logging for generator losses

Records every csv_log_steps:
- generator_loss, rec_loss, kl_loss, wavelet_loss
- Flushes to disk immediately (prevents data loss)
- Error handling (non-blocking)"
```

---

## Task 7: Add CSV Logging in Training Loop (Discriminator)

**Files:**
- Modify: `train_image_ddp.py:497-500` (in discriminator loss logging section)

**Step 1: Add CSV logging after WandB log**

After line 500 (after discriminator_loss wandb.log), add:

```python

                    # Log to CSV
                    if current_step % args.csv_log_steps == 0:
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
```

**Step 2: Verify indentation**

Ensure proper indentation (inside the `if global_rank == 0 and current_step % args.log_steps == 0:` block).

**Step 3: Commit**

```bash
git add train_image_ddp.py
git commit -m "feat: add CSV logging for discriminator loss

Records discriminator_loss every csv_log_steps.
Complements generator loss logging."
```

---

## Task 8: Add CSV Logging for Validation Metrics

**Files:**
- Modify: `train_image_ddp.py:540-541` (in validation function)

**Step 1: Add CSV logging after LPIPS wandb log**

After line 540 (after the lpips wandb.log), add:

```python

                    # Log validation metrics to CSV
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
```

**Step 2: Verify context**

Ensure this is inside the `if global_rank == 0:` block of the valid_model function.

**Step 3: Commit**

```bash
git add train_image_ddp.py
git commit -m "feat: add CSV logging for validation metrics

Records PSNR and LPIPS during validation.
Allows correlation between training losses and reconstruction quality."
```

---

## Task 9: Add Periodic Plot Generation at Checkpoints

**Files:**
- Modify: `train_image_ddp.py:571` (after checkpoint save)

**Step 1: Add plot generation**

After line 571 (`logger.info(f"Checkpoint has been saved to {file_path}.")`), add:

```python

                # Generate plot at checkpoint
                if not args.disable_plot:
                    try:
                        plot_training_curves(
                            csv_path=ckpt_dir / "training_losses.csv",
                            output_path=ckpt_dir / "training_curves.png",
                            disc_start=args.disc_start,
                            logger=logger
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update plot: {e}")
```

**Step 2: Verify placement**

Ensure this is inside the `if current_step % args.save_ckpt_step == 0 and global_rank == 0:` block.

**Step 3: Commit**

```bash
git add train_image_ddp.py
git commit -m "feat: generate plot at each checkpoint

Updates training_curves.png every save_ckpt_step.
Enables real-time progress monitoring without waiting for training completion."
```

---

## Task 10: Add Final Plot Generation and Cleanup

**Files:**
- Modify: `train_image_ddp.py:573-574` (before dist.destroy_process_group())

**Step 1: Add final cleanup and plot generation**

Before line 574 (`dist.destroy_process_group()`), add:

```python
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
                    disc_start=args.disc_start,
                    logger=logger
                )
            except Exception as e:
                logger.error(f"Failed to generate final plot: {e}")

```

**Step 2: Verify placement**

Ensure this code is after the training loop but before `dist.destroy_process_group()`.

**Step 3: Commit**

```bash
git add train_image_ddp.py
git commit -m "feat: add final plot generation and CSV cleanup

On normal training completion:
- Closes CSV file properly
- Generates training_curves_final.png
- Distinguishes from interrupted/checkpoint plots"
```

---

## Task 11: Update Dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add plotting dependencies**

Add to the end of `requirements.txt`:
```
pandas>=1.3.0
matplotlib>=3.3.0
scipy>=1.7.0
```

**Step 2: Verify additions**

Read `requirements.txt` to confirm the three packages are added.

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add pandas, matplotlib, scipy for plot generation

Required for training_curves visualization:
- pandas: CSV reading
- matplotlib: plot generation
- scipy: smoothing filter"
```

---

## Task 12: Update CLAUDE.md Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update discriminator start parameter in table**

Find the "Key Training Parameters" table (around line 115) and update:
```markdown
| `--disc_start` | 5000 | 0 |
```

To:
```markdown
| `--disc_start` | 20000 | 0 |
```

**Step 2: Add new parameters to documentation**

After the table, add a new section:

```markdown

### Loss Logging Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv_log_steps` | 50 | Log losses to CSV every N steps |
| `--disable_plot` | False | Disable automatic plot generation |

**CSV Output**: Training losses are automatically logged to `{ckpt_dir}/training_losses.csv`

**Plot Files**:
- `training_curves.png` - Updated at each checkpoint
- `training_curves_final.png` - Generated when training completes normally
- `training_curves_interrupted.png` - Generated if training is interrupted (Ctrl+C)

**Plot Features**:
- 3×3 multi-subplot layout (7 metrics)
- Smoothed curves + raw data overlay
- Discriminator start marker (vertical red line at 20000 steps)
- High resolution (300 DPI)
```

**Step 3: Verify documentation**

Read back the updated sections to ensure they're correct.

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update training parameters and add loss logging info

- Update disc_start default: 5000 → 20000
- Document csv_log_steps and disable_plot parameters
- Explain CSV output and plot file naming
- Describe visualization features"
```

---

## Task 13: Integration Test (Manual Verification)

**Files:**
- Test: `train_image_ddp.py`

**Step 1: Verify all imports**

Check that the following imports are present at the top of `train_image_ddp.py`:
- `import csv`
- `import signal`
- `import sys`

**Step 2: Verify argument parsing**

Check that these arguments exist:
- `--disc_start` with default=20000
- `--csv_log_steps` with default=50
- `--disable_plot` with action="store_true"

**Step 3: Verify plot function exists**

Confirm `plot_training_curves()` function is defined with correct signature.

**Step 4: Verify CSV initialization**

Confirm CSV writer initialization code exists after logger setup.

**Step 5: Verify signal handlers**

Confirm signal handler setup exists.

**Step 6: Verify logging points**

Confirm CSV logging exists at:
- Generator loss section
- Discriminator loss section
- Validation section

**Step 7: Verify checkpoint plot generation**

Confirm plot generation call exists after checkpoint save.

**Step 8: Verify final cleanup**

Confirm CSV close and final plot generation exist before `dist.destroy_process_group()`.

**Step 9: Create verification checklist**

All checks passed? Document any issues found.

**Step 10: Final commit**

```bash
git add -A
git commit -m "chore: verify integration of loss logging system

All components verified:
✓ Imports
✓ Arguments
✓ Plot function
✓ CSV initialization
✓ Signal handlers
✓ Logging points
✓ Checkpoint plots
✓ Final cleanup"
```

---

## Post-Implementation Validation

**Test commands** (to be run after implementation):

1. **Verify imports don't fail:**
```bash
python -c "from train_image_ddp import plot_training_curves; print('OK')"
```

2. **Check help text:**
```bash
python train_image_ddp.py --help | grep -E "(disc_start|csv_log_steps|disable_plot)"
```

3. **Test plot function directly** (create test CSV):
```bash
python -c "
import pandas as pd
from pathlib import Path
df = pd.DataFrame({
    'step': [0, 100, 200],
    'generator_loss': [1.0, 0.8, 0.6],
    'rec_loss': [0.5, 0.4, 0.3],
})
df.to_csv('test.csv', index=False)

from train_image_ddp import plot_training_curves
plot_training_curves(Path('test.csv'), Path('test_plot.png'), disc_start=100)
print('Plot generated: test_plot.png')
"
```

Expected: `test_plot.png` created successfully.

4. **Dry-run training** (if resources available):
```bash
torchrun --nproc_per_node=1 train_image_ddp.py \
    --exp_name test_logging \
    --max_steps 10 \
    --model_name WFIVAE2 \
    --model_config examples/wfivae2-image.json \
    --csv_log_steps 5
```

Expected:
- `results/test_logging/training_losses.csv` created
- CSV contains data
- Signal handler responds to Ctrl+C

---

## Summary

**Total Tasks**: 13
**Estimated Time**: 45-60 minutes
**Key Risks**:
- Line number drift (mitigate by searching for context strings)
- Indentation errors (verify with read-back)
- Missing error handling (each task includes try-except)

**Testing Strategy**:
- Incremental commits allow git bisect if issues arise
- Manual verification checklist (Task 13)
- Post-implementation validation commands

**Dependencies**:
- No breaking changes to existing code
- All new features are opt-in (via flags)
- Backward compatible with existing checkpoints

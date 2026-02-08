#!/bin/bash

# Commit script for loss logging and visualization feature
# Execute this script to commit all changes

cd "$(dirname "$0")"

echo "=== Committing Loss Logging and Visualization Feature ==="

# Add all modified files
git add train_image_ddp.py
git add requirements.txt
git add CLAUDE.md

# Create comprehensive commit
git commit -m "feat: implement CSV loss logging and visualization system

Major features:
- Update discriminator start parameter from 5000 to 20000 steps
- Add CSV logging for all training losses (every 50 steps by default)
- Implement plot_training_curves() with 3x3 multi-subplot visualization
- Add signal handlers for graceful interruption (Ctrl+C)
- Generate plots at checkpoints and training completion
- Support interrupted/final plot variants

Technical details:
- CSV fields: step, generator_loss, discriminator_loss, rec_loss, kl_loss, wavelet_loss, psnr, lpips
- Plot features: smoothing, discriminator start marker, high-res (300 DPI)
- Error handling: CSV I/O failures, plot generation errors, signal handling
- Distributed training safe: all I/O on rank 0 only

New parameters:
- --disc_start: default 20000 (was 5000)
- --csv_log_steps: default 50
- --disable_plot: disable automatic plot generation

Dependencies added:
- pandas>=1.3.0
- matplotlib>=3.3.0
- scipy>=1.7.0

Files modified:
- train_image_ddp.py: core implementation (200+ lines added)
- requirements.txt: add plotting dependencies
- CLAUDE.md: update documentation

Implements design from docs/plans/2026-02-01-training-loss-logging-design.md"

echo ""
echo "=== Commit completed successfully ==="
echo ""
echo "Next steps:"
echo "1. Review the commit: git show HEAD"
echo "2. Optionally create a tag: git tag -a v1.0-loss-logging -m 'Loss logging feature'"
echo "3. Push changes (if ready): git push origin master"

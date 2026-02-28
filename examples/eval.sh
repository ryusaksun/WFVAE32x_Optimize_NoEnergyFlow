#!/bin/bash
# NOTE: This is a legacy script from the video VAE project.
# scripts/eval.py does not exist in the current image-only codebase.
# For image VAE evaluation, use the validation built into train_image_ddp.py
# (--eval_steps, --eval_lpips, --eval_subset_size).

echo "ERROR: scripts/eval.py does not exist in this project."
echo "Use the built-in validation in train_image_ddp.py instead."
exit 1

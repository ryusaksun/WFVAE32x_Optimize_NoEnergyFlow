#!/bin/bash
# Quick Start Script for 1024x1024 Image VAE Training
# 快速启动脚本 - 1024分辨率图像VAE训练

set -e  # Exit on error

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate wfvae

echo "=========================================="
echo "WF-VAE Image Training - Quick Start"
echo "Using Manifest Files (Fast Loading)"
echo "=========================================="

# Check if manifest files exist
if [ ! -f "/mnt/sda/datasets/imagevae_1024/train_manifest.jsonl" ]; then
    echo "❌ Error: Training manifest not found at /mnt/sda/datasets/imagevae_1024/train_manifest.jsonl"
    exit 1
fi

if [ ! -f "/mnt/sda/datasets/imagevae_1024/eval_manifest.jsonl" ]; then
    echo "❌ Error: Validation manifest not found at /mnt/sda/datasets/imagevae_1024/eval_manifest.jsonl"
    exit 1
fi

echo "✓ Training manifest: /mnt/sda/datasets/imagevae_1024/train_manifest.jsonl"
echo "✓ Validation manifest: /mnt/sda/datasets/imagevae_1024/eval_manifest.jsonl"

# Create output directory
OUTPUT_DIR="/mnt/sdc/yyy_WFVAE_原版"
mkdir -p ${OUTPUT_DIR}
echo "✓ Output directory: ${OUTPUT_DIR}"

# Check GPU availability
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "✓ Detected ${NUM_GPUS} GPUs"

# Ask user for number of GPUs to use
read -p "Enter number of GPUs to use (default: ${NUM_GPUS}): " USER_GPUS
USER_GPUS=${USER_GPUS:-${NUM_GPUS}}

if [ ${USER_GPUS} -gt ${NUM_GPUS} ]; then
    echo "❌ Error: Requested ${USER_GPUS} GPUs but only ${NUM_GPUS} available"
    exit 1
fi

echo "✓ Using ${USER_GPUS} GPUs"

# Calculate batch size per GPU (adjust based on GPU memory)
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * USER_GPUS))
echo "✓ Total batch size: ${TOTAL_BATCH_SIZE} (${BATCH_SIZE_PER_GPU} per GPU)"

# Set environment variables
unset https_proxy
export WANDB_PROJECT=WFIVAE_1024
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((USER_GPUS-1)))
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=========================================="
echo "Starting training with Manifest files..."
echo "Max steps: 15000"
echo "Checkpoint every: 1000 steps"
echo "Validation every: 500 steps"
echo "=========================================="

# Launch training with manifest files
torchrun \
    --nnodes=1 \
    --nproc_per_node=${USER_GPUS} \
    --master_addr=localhost \
    --master_port=12136 \
    train_image_ddp.py \
    --exp_name WFIVAE_1024 \
    --image_path /mnt/sda/datasets/imagevae_1024/train_manifest.jsonl \
    --eval_image_path /mnt/sda/datasets/imagevae_1024/eval_manifest.jsonl \
    --use_manifest \
    --ckpt_dir ${OUTPUT_DIR} \
    --model_name WFIVAE2 \
    --model_config examples/wfivae2-image-1024.json \
    --resolution 1024 \
    --batch_size ${BATCH_SIZE_PER_GPU} \
    --lr 1e-5 \
    --weight_decay 1e-4 \
    --epochs 10 \
    --disc_start 5 \
    --disc_weight 0.5 \
    --kl_weight 1e-6 \
    --perceptual_weight 1.0 \
    --loss_type l1 \
    --disc_cls causalimagevae.model.losses.LPIPSWithDiscriminator \
    --save_ckpt_step 1000 \
    --eval_steps 500 \
    --max_steps 15000 \
    --eval_batch_size 4 \
    --eval_subset_size 500 \
    --eval_num_image_log 8 \
    --eval_lpips \
    --ema \
    --ema_decay 0.999 \
    --mix_precision bf16 \
    --log_steps 10 \
    --dataset_num_worker 4 \
    --wavelet_loss \
    --wavelet_weight 0.1 \
    --find_unused_parameters

echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "=========================================="

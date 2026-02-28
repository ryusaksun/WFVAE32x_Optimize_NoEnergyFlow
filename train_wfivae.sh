#!/bin/bash

# ============================================
# WFIVAE2 (WF-VAE with Energy Flow) 训练脚本
# ============================================
#
# 使用方法：
#
# 1. 从头开始训练（默认使用 GPU 0）：
#    bash train_wfivae.sh
#
# 2. 指定单个 GPU：
#    GPU=3 bash train_wfivae.sh
#
# 3. 多卡 DDP 训练（自动检测并使用 torchrun）：
#    GPU=0,1,2,3 bash train_wfivae.sh
#    GPU=0,1 bash train_wfivae.sh
#
# 4. 从checkpoint恢复训练：
#    RESUME_CKPT="/path/to/checkpoint.ckpt" bash train_wfivae.sh
#
# 5. 组合使用：
#    GPU=0,1,2,3 RESUME_CKPT="/path/to/ckpt" bash train_wfivae.sh
#
# 6. 指定训练/验证集划分比例（默认 0.9）：
#    TRAIN_RATIO=0.8 bash train_wfivae.sh
#
# 7. 指定分辨率（512 或 1024，默认 1024）：
#    RESOLUTION=512 bash train_wfivae.sh
#
# 8. 使用完整验证集输出 PatchGAN patch 分数（默认开启）：
#    EVAL_SUBSET_SIZE=0 bash train_wfivae.sh
#
# 9. 覆盖训练步频参数：
#    EVAL_STEPS=500 SAVE_CKPT_STEP=1000 EPOCHS=500 bash train_wfivae.sh
#
# ============================================

set -e  # 遇到错误立即退出

# ============================================
# Conda 环境激活（根据需要取消注释）
# ============================================
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate wfvae

# ============================================
# GPU 配置
# ============================================

# 设置GPU (默认使用 GPU 0，可通过 GPU 环境变量覆盖)
GPU=${GPU:-6}
export CUDA_VISIBLE_DEVICES=$GPU

# 计算 GPU 数量 (用于判断是否使用 DDP)
NUM_GPUS=$(echo "$GPU" | tr ',' '\n' | wc -l | tr -d ' ')
echo "检测到 $NUM_GPUS 个 GPU: $GPU"

# ============================================
# NCCL 配置（多卡训练时可能需要）
# ============================================
# export GLOO_SOCKET_IFNAME=bond0
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TC=162
# export NCCL_IB_TIMEOUT=22
# export NCCL_PXN_DISABLE=0
# export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================
# WandB 配置
# ============================================
export WANDB_PROJECT="${WANDB_PROJECT:-WFIVAE}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"  # 1/true/yes: 关闭 wandb；0/false/no: 开启
DISABLE_WANDB="$(echo "$DISABLE_WANDB" | tr '[:upper:]' '[:lower:]')"
if [ "$DISABLE_WANDB" = "1" ] || [ "$DISABLE_WANDB" = "true" ] || [ "$DISABLE_WANDB" = "yes" ]; then
    export WANDB_MODE=disabled  # 彻底禁止 wandb 产生本地文件
    WANDB_ARGS="--disable_wandb"
    WANDB_STATUS="关闭"
else
    WANDB_ARGS=""
    WANDB_STATUS="开启 (project: ${WANDB_PROJECT})"
fi

# ============================================
# 路径配置
# ============================================
#
# 损失日志功能 (自动启用):
# - CSV 日志: ${OUTPUT_DIR}/${EXP_NAME}/${EXP_NAME}.csv
# - 训练曲线图: 每次验证后自动更新
# - PatchGAN patch 分数: ${OUTPUT_DIR}/${EXP_NAME}/val_patch_scores/step_xxxxxxxx(_ema)/
# - 使用 --csv_log_steps 调整日志频率
# - 使用 --disable_plot 禁用自动绘图
#

# 原始数据清单文件（JSONL格式，支持 image_path/path/target 字段）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$SCRIPT_DIR")"
DEFAULT_MANIFEST="${SCRIPT_DIR}/ssk_image_manifest.jsonl"
ORIGINAL_MANIFEST="${ORIGINAL_MANIFEST:-$DEFAULT_MANIFEST}"

# 输出目录 (默认放在 /mnt/sdc 下，包含项目名以区分实验)
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/sdb/${PROJECT_NAME}}"

# 临时划分文件路径 (放在输出目录下)
TRAIN_MANIFEST="${OUTPUT_DIR}/train_manifest.jsonl"
EVAL_MANIFEST="${OUTPUT_DIR}/eval_manifest.jsonl"

# 分辨率配置 (默认 1024)
RESOLUTION="${RESOLUTION:-256}"

# 训练/验证步频与日志配置（可通过环境变量覆盖）
EPOCHS="${EPOCHS:-1000}"
SAVE_CKPT_STEP="${SAVE_CKPT_STEP:-2000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
EVAL_SUBSET_SIZE="${EVAL_SUBSET_SIZE:-30}"     # 验证子集大小，0 表示使用完整验证集
EVAL_NUM_IMAGE_LOG="${EVAL_NUM_IMAGE_LOG:-20}"  # 验证重建图与 patch 可视化样本数量（默认保持一致）
CSV_LOG_STEPS="${CSV_LOG_STEPS:-50}"
LOG_STEPS="${LOG_STEPS:-10}"
DATASET_NUM_WORKER="${DATASET_NUM_WORKER:-8}"

# 根据分辨率选择默认配置 (EXP_NAME 包含 disc5 标识)
if [ "$RESOLUTION" = "1024" ]; then
    DEFAULT_MODEL_CONFIG="examples/wfivae2-image-1024.json"
    DEFAULT_EXP_NAME="$PROJECT_NAME"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
elif [ "$RESOLUTION" = "256" ]; then
    DEFAULT_MODEL_CONFIG="examples/wfivae2-image-1024.json"
    DEFAULT_EXP_NAME="$PROJECT_NAME"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
else
    # 512 或其他分辨率，模型配置通用（无分辨率参数）
    DEFAULT_MODEL_CONFIG="examples/wfivae2-image-1024.json"
    DEFAULT_EXP_NAME="$PROJECT_NAME"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
fi

MODEL_CONFIG="${MODEL_CONFIG:-$DEFAULT_MODEL_CONFIG}"
EXP_NAME="${EXP_NAME:-$DEFAULT_EXP_NAME}"

# 训练集比例 (默认 0.9)
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"

# Resume设置（可选）
RESUME_CKPT="${RESUME_CKPT:-}"

# 判别器刷新设置
REFRESH_DISC="${REFRESH_DISC:-}"                   # 非空则恢复时刷新判别器
DISC_REFRESH_EVERY="${DISC_REFRESH_EVERY:-0}"      # 每N步刷新判别器（0=禁用）
DISC_REFRESH_WARMUP="${DISC_REFRESH_WARMUP:-100}"  # 刷新后判别器独占训练步数

# 判别器学习率 (TTUR: Two Time-scale Update Rule)
DISC_LR="${DISC_LR:-4e-5}"                             # 判别器学习率，默认4倍于生成器(1e-5)

# 自适应权重 clamp 上限（控制判别器对生成器的最大影响力）
# 默认 1e6（原始值），降低可减少判别器过强导致的伪影
ADAPTIVE_WEIGHT_CLAMP="${ADAPTIVE_WEIGHT_CLAMP:-1e5}"

# 训练精度（bf16/fp16/fp32，fp32 对判别器更稳定但速度慢 40-60%）
MIX_PRECISION="${MIX_PRECISION:-fp32}"

# 学习 logvar（自动平衡重建损失与 GAN 损失的量级）
LEARN_LOGVAR="${LEARN_LOGVAR:-1}"                        # 非空启用，置空禁用
LOGVAR_LR="${LOGVAR_LR:-1e-2}"                            # logvar 独立学习率（快速找到均衡点）

# R1 梯度惩罚（防止判别器过拟合，0=禁用）
R1_WEIGHT="${R1_WEIGHT:-0}"
R1_START="${R1_START:-0}"                               # R1 开始生效的 step（0=从头开始）
R1_WARMUP_STEPS="${R1_WARMUP_STEPS:-0}"                 # R1 线性 warmup 步数（0=立即满值）

# 判别器自动刷新（停滞检测）
DISC_AUTO_REFRESH="${DISC_AUTO_REFRESH:-}"            # 非空启用自动刷新
DISC_STALE_WINDOW="${DISC_STALE_WINDOW:-500}"         # 滚动窗口大小（判别器步数）
DISC_STALE_LOSS="${DISC_STALE_LOSS:-0.8}"             # disc_loss 均值阈值
DISC_STALE_STD="${DISC_STALE_STD:-0.02}"              # disc_loss 标准差阈值
DISC_AUTO_COOLDOWN="${DISC_AUTO_COOLDOWN:-5000}"       # 自动刷新冷却步数

# ============================================
# 创建输出目录
# ============================================

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "创建输出目录: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# ============================================
# 数据集划分函数
# ============================================

split_dataset() {
    echo "================================================"
    echo "划分数据集 (训练集: ${TRAIN_RATIO}, 验证集: $(echo "1 - $TRAIN_RATIO" | bc))"
    echo "================================================"

    python3 << EOF
import random, sys

# 读取原始数据
with open("$ORIGINAL_MANIFEST", "r", encoding="utf-8") as f:
    lines = f.readlines()

total = len(lines)
if total == 0:
    print("错误: manifest 文件为空，无法划分数据集!")
    sys.exit(1)
print(f"总样本数: {total}")

# 设置随机种子确保可复现
random.seed(42)

# 随机打乱索引
indices = list(range(total))
random.shuffle(indices)

# 划分
split_idx = int(total * $TRAIN_RATIO)
train_indices = indices[:split_idx]
eval_indices = indices[split_idx:]

train_lines = [lines[i] for i in train_indices]
eval_lines = [lines[i] for i in eval_indices]

# 保存训练集
with open("$TRAIN_MANIFEST", "w", encoding="utf-8") as f:
    f.writelines(train_lines)
print(f"训练集: {len(train_lines)} 条 -> $TRAIN_MANIFEST")

# 保存验证集
with open("$EVAL_MANIFEST", "w", encoding="utf-8") as f:
    f.writelines(eval_lines)
print(f"验证集: {len(eval_lines)} 条 -> $EVAL_MANIFEST")

print(f"划分完成! 训练集 {len(train_lines)/total*100:.1f}%, 验证集 {len(eval_lines)/total*100:.1f}%")
EOF
}

# ============================================
# 清理函数 (训练结束后调用)
# ============================================

cleanup() {
    echo ""
    echo "================================================"
    echo "清理临时数据集划分文件..."
    echo "================================================"

    if [ -f "$TRAIN_MANIFEST" ]; then
        rm -f "$TRAIN_MANIFEST"
        echo "已删除: $TRAIN_MANIFEST"
    fi

    if [ -f "$EVAL_MANIFEST" ]; then
        rm -f "$EVAL_MANIFEST"
        echo "已删除: $EVAL_MANIFEST"
    fi

    echo "数据集已还原 (原始文件未修改)"
    echo "================================================"
}

# 注册清理函数，无论脚本如何退出都会执行
trap cleanup EXIT

# ============================================
# 显示配置信息
# ============================================

echo "================================================"
echo "WFIVAE2 Training (WF-VAE with Energy Flow)"
echo "================================================"
echo "原始数据集: $ORIGINAL_MANIFEST"
echo "输出目录: $OUTPUT_DIR"
echo "模型配置: $MODEL_CONFIG"
echo "实验名称: $EXP_NAME"
echo "分辨率: ${RESOLUTION}x${RESOLUTION}"
echo "训练集比例: $TRAIN_RATIO"
echo "GPU: $GPU ($NUM_GPUS 卡)"
echo "训练 Epochs: $EPOCHS"
echo "Checkpoint间隔: $SAVE_CKPT_STEP"
echo "验证间隔: $EVAL_STEPS"
echo "学习率: gen=1e-5, disc=$DISC_LR (TTUR)"
echo "自适应权重 clamp: $ADAPTIVE_WEIGHT_CLAMP"
echo "训练精度: $MIX_PRECISION"
echo "Learn logvar: ${LEARN_LOGVAR:-禁用} (lr=$LOGVAR_LR)"
echo "R1 梯度惩罚: $R1_WEIGHT (start=$R1_START, warmup=$R1_WARMUP_STEPS)"
echo "WandB: $WANDB_STATUS"
if [ "$EVAL_SUBSET_SIZE" = "0" ]; then
    echo "验证样本数: 全量"
else
    echo "验证样本数: $EVAL_SUBSET_SIZE"
fi
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "训练模式: DDP 多卡并行 (torchrun)"
else
    echo "训练模式: 单卡训练"
fi
echo "================================================"
echo "架构特点:"
echo "  - WF-VAE: Wavelet-driven energy Flow VAE"
echo "  - Multi-level Haar Wavelet Transform"
echo "  - Energy flow pathway for frequency info"
echo "  - 压缩比: 8倍"
LATENT_H=$((RESOLUTION / 8))
echo "  - 潜变量: [B, latent_dim, ${LATENT_H}, ${LATENT_H}]"
echo "================================================"

# 显示resume信息
if [ -n "$RESUME_CKPT" ]; then
    echo "RESUME MODE: ON"
    echo "Checkpoint: $RESUME_CKPT"
    if [ -n "$REFRESH_DISC" ]; then
        echo "判别器刷新: ON (恢复时重新初始化)"
    fi
    echo "================================================"
else
    echo "RESUME MODE: OFF (从头开始训练)"
    echo "================================================"
fi
if [ "$DISC_REFRESH_EVERY" -gt 0 ] 2>/dev/null; then
    echo "判别器定期刷新: 每 ${DISC_REFRESH_EVERY} 步, warmup ${DISC_REFRESH_WARMUP} 步"
    echo "================================================"
fi
if [ -n "$DISC_AUTO_REFRESH" ]; then
    echo "判别器自动刷新: ON (window=${DISC_STALE_WINDOW}, loss>${DISC_STALE_LOSS}, std<${DISC_STALE_STD}, cooldown=${DISC_AUTO_COOLDOWN})"
    echo "================================================"
fi

# ============================================
# 检查文件
# ============================================

if [ ! -f "$ORIGINAL_MANIFEST" ]; then
    echo "Error: 原始数据清单文件不存在: $ORIGINAL_MANIFEST"
    echo "请设置 ORIGINAL_MANIFEST 环境变量为你的数据清单路径"
    echo "例如: ORIGINAL_MANIFEST=/path/to/manifest.jsonl bash train_wfivae.sh"
    exit 1
fi

if [ ! -f "$MODEL_CONFIG" ]; then
    echo "Error: 模型配置文件不存在: $MODEL_CONFIG"
    exit 1
fi

# ============================================
# 执行数据集划分
# ============================================

split_dataset

# ============================================
# 开始训练
# ============================================

echo ""
echo "Starting training..."
echo ""

# 日志文件 (放在输出目录下)
LOG_FILE="${OUTPUT_DIR}/training_wfivae_${RESOLUTION}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# 构建resume参数
RESUME_ARGS=""
if [ -n "$RESUME_CKPT" ]; then
    RESUME_ARGS="--resume_from_checkpoint $RESUME_CKPT"
fi

# 训练参数
TRAIN_ARGS="train_image_ddp.py \
    --exp_name $EXP_NAME \
    --image_path $TRAIN_MANIFEST \
    --eval_image_path $EVAL_MANIFEST \
    --use_manifest \
    --model_name WFIVAE2 \
    --model_config $MODEL_CONFIG \
    --ckpt_dir $OUTPUT_DIR \
    --resolution $RESOLUTION \
    --batch_size $BATCH_SIZE \
    --lr 1e-5 \
    --weight_decay 1e-4 \
    --epochs $EPOCHS \
    --save_ckpt_step $SAVE_CKPT_STEP \
    --eval_steps $EVAL_STEPS \
    --eval_resolution $RESOLUTION \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --eval_subset_size $EVAL_SUBSET_SIZE \
    --eval_num_image_log $EVAL_NUM_IMAGE_LOG \
    --eval_lpips \
    --mix_precision $MIX_PRECISION \
    --disc_cls causalimagevae.model.losses.LPIPSWithDiscriminator \
    --disc_start 5 \
    --disc_weight 0.5 \
    --disc_lr $DISC_LR \
    --adaptive_weight_clamp $ADAPTIVE_WEIGHT_CLAMP \
    --r1_weight $R1_WEIGHT \
    --r1_start $R1_START \
    --r1_warmup_steps $R1_WARMUP_STEPS \
    --kl_weight 1e-6 \
    --perceptual_weight 1.0 \
    --loss_type l1 \
    --wavelet_loss \
    --wavelet_weight 0.1 \
    --ema \
    --ema_decay 0.999 \
    --csv_log_steps $CSV_LOG_STEPS \
    --log_steps $LOG_STEPS \
    --dataset_num_worker $DATASET_NUM_WORKER \
    $WANDB_ARGS \
    --find_unused_parameters \
    --disc_refresh_every $DISC_REFRESH_EVERY \
    --disc_refresh_warmup $DISC_REFRESH_WARMUP \
    ${REFRESH_DISC:+--refresh_discriminator} \
    ${LEARN_LOGVAR:+--learn_logvar} \
    --logvar_lr $LOGVAR_LR \
    ${DISC_AUTO_REFRESH:+--disc_auto_refresh} \
    --disc_stale_window $DISC_STALE_WINDOW \
    --disc_stale_loss_threshold $DISC_STALE_LOSS \
    --disc_stale_std_threshold $DISC_STALE_STD \
    --disc_auto_refresh_cooldown $DISC_AUTO_COOLDOWN \
    $RESUME_ARGS"

# 选择可用端口 (29500-29599)
if [ -z "$MASTER_PORT" ]; then
    for _port in $(shuf -i 29500-29599 -n 100 2>/dev/null || seq 29500 29599); do
        if ! lsof -i :"$_port" -sTCP:LISTEN >/dev/null 2>&1; then
            MASTER_PORT=$_port
            break
        fi
    done
    MASTER_PORT=${MASTER_PORT:-29500}  # fallback
fi
echo "使用 torchrun 启动训练 (${NUM_GPUS} GPUs, port: ${MASTER_PORT})..."

# 始终使用 torchrun（train_image_ddp.py 需要 DDP 环境）
stdbuf -oL torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    $TRAIN_ARGS

echo ""
echo "================================================"
echo "Training completed!"
echo "输出目录: $OUTPUT_DIR"
echo "================================================"

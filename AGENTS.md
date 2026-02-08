# AGENTS.md

本文件面向在本仓库协作的工程师/Agent，目标是快速对齐项目现状与高频工作流。

## 1. 语言与协作约定

- 默认使用中文（简体）沟通。
- 变更训练逻辑后，优先做最小可运行验证（语法检查 + 一次短流程验证）。
- 不要在未经确认的情况下修改数据路径为你本地私有路径。

## 2. 项目定位

- 这是一个 **图像版 WF-VAE** 训练仓库（核心模型：`WFIVAE2`）。
- 主训练入口：
  - `/Users/ryuichi/Desktop/renxing/WFVAE-series/WF-VAE-8x_disc_start_5/train_wfivae.sh`（推荐）
  - `/Users/ryuichi/Desktop/renxing/WFVAE-series/WF-VAE-8x_disc_start_5/train_image_ddp.py`（底层训练器）
- 模型注册：
  - `WFIVAE2` 在 `/Users/ryuichi/Desktop/renxing/WFVAE-series/WF-VAE-8x_disc_start_5/causalimagevae/model/vae/modeling_wfvae2.py` 注册到 `ModelRegistry`

## 3. 关键目录结构

- `causalimagevae/model/`：模型、损失、模块与工具
- `causalimagevae/dataset/`：普通目录数据集与 manifest 数据集
- `examples/`：模型配置与示例脚本（当前仅有 `wfivae2-image-1024.json`）
- `scripts/`：重建、测试、权重转换等工具
- `train_image_ddp.py`：DDP 训练、验证、日志、checkpoint 主逻辑
- `train_wfivae.sh`：数据划分 + 启动 torchrun 的生产脚本

## 4. 当前训练产物（重要）

在 `{ckpt_dir}/{exp_name-lr...-bs...-rs...}/` 下：

- `training_losses.csv`：训练/验证指标 CSV
- `training_curves.png`：训练曲线（每次验证后更新）
- `training_curves_final.png`：训练结束最终图
- `val_images/original/` 与 `val_images/reconstructed/`：验证图像
- `val_patch_scores/step_xxxxxxxx/` 与 `val_patch_scores/step_xxxxxxxx_ema/`：
  - `real_logits.npy`
  - `recon_logits.npy`
  - `real_sigmoid.npy`
  - `recon_sigmoid.npy`
  - `summary.csv`
- `checkpoint-*.ckpt`：模型与优化器状态

## 5. 训练脚本推荐用法

默认启动：

```bash
cd /Users/ryuichi/Desktop/renxing/WFVAE-series/WF-VAE-8x_disc_start_5
bash train_wfivae.sh
```

高频环境变量（脚本已支持）：

```bash
GPU=0,1,2,3 \
ORIGINAL_MANIFEST=/path/to/all_images.jsonl \
OUTPUT_DIR=/path/to/output \
RESOLUTION=1024 \
EVAL_SUBSET_SIZE=0 \
MAX_STEPS=100000 \
EVAL_STEPS=1000 \
SAVE_CKPT_STEP=2000 \
bash train_wfivae.sh
```

说明：

- `EVAL_SUBSET_SIZE=0` 表示验证集全量（用于完整 patch 分数导出）。
- `MAX_STEPS` 现在在训练器里会真实触发停训（不再只是进度条参数）。
- 若 `RESOLUTION=512` 且未提供专用配置，脚本会默认回退到 `examples/wfivae2-image-1024.json`；如有 512 专用配置，建议显式传入 `MODEL_CONFIG=...`。

## 6. 代码修改高风险点

- `train_image_ddp.py` 的 DDP 汇总逻辑使用 `all_gather_object`，修改返回结构时要同步更新 gather/日志/保存链路。
- patch 分数导出依赖验证 batch 中的 `index` 字段；`ValidImageDataset` 与 `ValidManifestImageDataset` 都已提供该字段，改数据集时不要丢。
- checkpoint 中判别器键名是历史拼写 `dics_model`，不要随意改名，否则旧 checkpoint 恢复会断。

## 7. 文档一致性提醒

仓库里有部分旧文档仍提到不存在的文件名（如 `train_image_ddp.sh`、`examples/wfivae2-image.json`）。  
若你做了脚本或流程升级，请优先同步：

- `/Users/ryuichi/Desktop/renxing/WFVAE-series/WF-VAE-8x_disc_start_5/README_image_training.md`
- `/Users/ryuichi/Desktop/renxing/WFVAE-series/WF-VAE-8x_disc_start_5/QUICK_REFERENCE.md`
- `/Users/ryuichi/Desktop/renxing/WFVAE-series/WF-VAE-8x_disc_start_5/CONFIG_1024.md`

## 8. 提交前最小检查

```bash
python3 -m py_compile train_image_ddp.py
bash -n train_wfivae.sh
```

如果改了数据集或验证输出，建议再做一次最小训练验证（跑到第一次 validation）确认以下目录生成正常：

- `val_patch_scores/step_xxxxxxxx/`
- `val_patch_scores/step_xxxxxxxx_ema/`
- `training_curves.png`

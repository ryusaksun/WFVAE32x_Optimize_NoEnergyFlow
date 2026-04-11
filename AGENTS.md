# AGENTS.md

本文件面向在本仓库协作的工程师/Agent，目标是快速对齐项目现状与高频工作流。

## 1. 语言与协作约定

- 默认使用中文（简体）沟通。
- 变更训练逻辑后，优先做最小可运行验证（语法检查 + 一次短流程验证）。
- 不要在未经确认的情况下修改数据路径为你本地私有路径。

## 2. 项目定位

- 这是一个 **图像版 WF-VAE** 训练仓库（核心模型：`WFIVAE2`）。
- 主训练入口：
  - `train_wfivae.sh`（推荐，生产入口）
  - `train_image_ddp.py`（底层训练器）
- 模型注册：
  - `WFIVAE2` 在 `causalimagevae/model/vae/modeling_wfvae2.py` 注册到 `ModelRegistry`

## 3. 关键目录结构

- `causalimagevae/model/`：模型、损失、模块与工具
- `causalimagevae/dataset/`：普通目录数据集与 manifest 数据集
- `examples/`：模型配置与示例脚本（当前常用配置是 `wfivae2-image-192bc.json`）
- `scripts/`：重建、测试、权重转换等工具
- `train_image_ddp.py`：DDP 训练、验证、日志、checkpoint 主逻辑
- `train_wfivae.sh`：manifest 选择/可选划分 + 启动 `torchrun` 的生产脚本

## 4. 当前训练产物（重要）

在 `{ckpt_dir}/{exp_name-lr...-bs...-rs...}/` 下：

- `{exp_name}.csv`：训练/验证指标 CSV
- `training_curves.png`：训练曲线（每次验证后更新）
- `training_curves_final.png`：训练结束最终图
- `val_images/original/` 与 `val_images/reconstructed/`：验证图像
- `checkpoints/checkpoint-*.ckpt`：模型、优化器、scaler、sampler、EMA 状态

当前代码**不会**导出 `val_patch_scores/`。仓库内仍有旧文档提到该目录，阅读时以代码行为为准。

## 5. 训练脚本推荐用法

默认启动：

```bash
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
GRAD_ACCUM_STEPS=4 \
EVAL_STEPS=1000 \
SAVE_CKPT_STEP=2000 \
bash train_wfivae.sh
```

说明：

- `EVAL_SUBSET_SIZE=0` 表示验证集全量。
- `MAX_STEPS` 会真实触发停训。
- `GRAD_ACCUM_STEPS` 会传入训练器，等效增大 batch size。
- 验证阶段当前使用 **deterministic posterior (`mode`)**，指标不再受随机采样影响。
- 验证 LPIPS 当前记录的是 **每张图像的标量 LPIPS**，不是 spatial heatmap。
- `train_wfivae.sh` 默认会在 `VAL_MANIFEST` 留空时按 `TRAIN_RATIO` 自动切分；若你已预先划分好验证集，显式传入 `VAL_MANIFEST=/path/to/val.jsonl` 即可。
- 若 `RESOLUTION=512` 且未提供专用配置，脚本当前默认仍会使用 `examples/wfivae2-image-32x-192bc.json`；如有专用配置，建议显式传入 `MODEL_CONFIG=...`。

## 6. 代码修改高风险点

- `train_image_ddp.py` 的验证汇总逻辑使用 `all_gather_object`，修改返回结构时要同步更新 gather/日志/保存链路。
- 验证集图像保存与指标去重依赖验证 batch 中的 `index` 字段；`ValidImageDataset` 与 `ValidManifestImageDataset` 都已提供该字段，改数据集时不要丢。
- 训练循环包含梯度累积、`no_sync()`、epoch 末尾 partial flush，以及手动梯度同步。若修改生成器 loss 路径、`learn_logvar`、或 DDP 包装边界，要同时检查 `aux_gen_params` 的同步逻辑。
- checkpoint 中保存的 `sampler_state` 是训练器按“已消费 batch 位置”构造的恢复状态；修改 checkpoint/save/resume 流程时，要同步检查 epoch 边界恢复是否会重跑样本。
- checkpoint 中判别器键名是历史拼写 `dics_model`，不要随意改名，否则旧 checkpoint 恢复会断。

## 7. 文档一致性提醒

仓库里有部分旧文档仍提到旧路径、旧文件名或旧产物（如 `train_image_ddp.sh`、`examples/wfivae2-image.json`、`training_losses.csv`、`val_patch_scores/`）。  
若你做了脚本或流程升级，请优先同步：

- `docs/README_image_training.md`
- `docs/QUICK_REFERENCE.md`
- 其他仍引用旧工作流的文档

## 8. 提交前最小检查

```bash
python3 -m py_compile train_image_ddp.py
bash -n train_wfivae.sh
```

如果改了数据集、验证输出、DDP/恢复逻辑，建议再做一次最小训练验证（跑到第一次 validation 或一次 resume）确认以下产物正常：

- `{exp_name}.csv`
- `val_images/original/`
- `val_images/reconstructed/`
- `checkpoints/checkpoint-*.ckpt`
- `training_curves.png`

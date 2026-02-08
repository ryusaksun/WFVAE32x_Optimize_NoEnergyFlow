# 使用 Manifest 文件训练

如果您希望使用 manifest 文件 (JSONL格式) 来加载数据，而不是自动扫描目录，可以按照以下步骤操作。

## Manifest 文件格式

您的 manifest 文件应该是 JSONL 格式（每行一个JSON对象）：

```jsonl
{"image_path": "train/img001.jpg", "caption": "可选的描述", "metadata": {"key": "value"}}
{"image_path": "train/img002.png", "caption": "可选的描述", "metadata": {"key": "value"}}
```

必需字段：
- `image_path`: 图像的相对路径或绝对路径

可选字段：
- `caption`: 图像描述（可用于条件生成）
- `metadata`: 其他元数据（字典）

## 修改训练脚本使用 Manifest

### 方法1: 修改 train_image_ddp.py

在 `train_image_ddp.py` 中找到数据加载部分（约283-300行），替换为：

```python
# 原代码：
# dataset = ImageDataset(
#     args.image_path,
#     resolution=args.resolution,
#     cache_file="image_cache.pkl",
#     is_main_process=global_rank == 0,
# )

# 新代码：
from causalimagevae.dataset import ManifestImageDataset, ValidManifestImageDataset

dataset = ManifestImageDataset(
    manifest_path=args.train_manifest,  # 新参数
    base_dir=args.image_path,           # 作为base_dir
    resolution=args.resolution,
)

# 同样修改验证集：
val_dataset = ValidManifestImageDataset(
    manifest_path=args.eval_manifest,   # 新参数
    base_dir=args.eval_image_path,      # 作为base_dir
    resolution=args.eval_resolution,
    crop_size=args.eval_resolution,
)
```

### 方法2: 添加命令行参数

在 `train_image_ddp.py` 的参数解析部分（约640行）添加：

```python
# Data
parser.add_argument("--image_path", type=str, default=None, help="path to training images")
parser.add_argument("--train_manifest", type=str, default=None, help="path to training manifest JSONL")  # 新增
parser.add_argument("--eval_manifest", type=str, default=None, help="path to eval manifest JSONL")      # 新增
```

然后在数据加载部分使用条件判断：

```python
if args.train_manifest:
    # 使用 manifest
    dataset = ManifestImageDataset(
        manifest_path=args.train_manifest,
        base_dir=args.image_path,
        resolution=args.resolution,
    )
else:
    # 使用目录扫描
    dataset = ImageDataset(
        args.image_path,
        resolution=args.resolution,
        cache_file="image_cache.pkl",
        is_main_process=global_rank == 0,
    )
```

## 使用 Manifest 启动训练

修改 `train_image_ddp.sh`，添加 manifest 参数：

```bash
torchrun \
    --nnodes=1 --nproc_per_node=8 \
    train_image_ddp.py \
    --exp_name WFIVAE_1024 \
    --image_path /mnt/sda/datasets/imagevae_1024/train \
    --eval_image_path /mnt/sda/datasets/imagevae_1024/eval \
    --train_manifest /mnt/sda/datasets/imagevae_1024/train_manifest.jsonl \
    --eval_manifest /mnt/sda/datasets/imagevae_1024/eval_manifest.jsonl \
    [其他参数...]
```

## 创建 Manifest 文件

如果您还没有 manifest 文件，可以使用以下脚本生成：

```python
import os
import json
from pathlib import Path

def create_manifest(image_dir, output_manifest, extensions=['.jpg', '.png', '.jpeg', '.webp']):
    """从图像目录创建 manifest 文件"""
    manifest_lines = []
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                # 获取相对路径
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, image_dir)
                
                # 创建 manifest 条目
                entry = {
                    "image_path": rel_path,
                    "caption": "",  # 可选
                    "metadata": {}  # 可选
                }
                manifest_lines.append(json.dumps(entry, ensure_ascii=False))
    
    # 写入文件
    with open(output_manifest, 'w', encoding='utf-8') as f:
        f.write('\n'.join(manifest_lines))
    
    print(f"Created manifest with {len(manifest_lines)} images: {output_manifest}")

# 使用示例
create_manifest(
    '/mnt/sda/datasets/imagevae_1024/train',
    '/mnt/sda/datasets/imagevae_1024/train_manifest.jsonl'
)

create_manifest(
    '/mnt/sda/datasets/imagevae_1024/eval',
    '/mnt/sda/datasets/imagevae_1024/eval_manifest.jsonl'
)
```

## Manifest 的优势

1. **灵活性**: 可以精确控制哪些图像被加载
2. **元数据**: 可以存储额外信息（标签、描述等）
3. **过滤**: 可以轻松过滤特定条件的图像
4. **版本控制**: manifest 文件易于版本管理
5. **调试**: 容易追踪和重现数据加载问题

## 测试 Manifest 数据加载

```python
from causalimagevae.dataset.manifest_dataset import ManifestImageDataset

# 测试加载
dataset = ManifestImageDataset(
    manifest_path='/mnt/sda/datasets/imagevae_1024/train_manifest.jsonl',
    base_dir='/mnt/sda/datasets/imagevae_1024',
    resolution=1024
)

print(f"Dataset size: {len(dataset)}")

# 测试第一个样本
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Image path: {sample['path']}")
print(f"Metadata: {sample['metadata']}")
```

## 注意事项

1. **路径格式**: manifest 中的路径可以是相对或绝对路径
2. **编码**: JSONL 文件使用 UTF-8 编码
3. **错误处理**: 如果某个图像加载失败，会自动跳到下一个
4. **性能**: manifest 加载通常比目录扫描更快

如果您需要使用 manifest 文件，我可以帮您修改 `train_image_ddp.py` 以支持这个功能。

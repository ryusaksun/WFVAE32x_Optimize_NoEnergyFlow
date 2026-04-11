"""
将 SA-1B 数据集从 1024px 缩放到 256px，并生成新的 manifest 文件。

用法:
    python resize_sa1b_256.py [--workers 16]

输入: /mnt/hpfs/HDU/ssk/SA-1B/sa1b_1024/*.jpg
输出: /mnt/hpfs/HDU/ssk/SA-1B_256/sa1b_256/*.jpg
      /mnt/hpfs/HDU/ssk/SA-1B_256/train_manifest.jsonl
      /mnt/hpfs/HDU/ssk/SA-1B_256/val_manifest.jsonl
"""

import os
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from PIL import Image

SRC_DIR = "/mnt/hpfs/HDU/ssk/SA-1B/sa1b_1024"
DST_DIR = "/mnt/hpfs/HDU/ssk/SA-1B_256/sa1b_256"
SRC_MANIFEST_DIR = "/mnt/hpfs/HDU/ssk/SA-1B"
DST_MANIFEST_DIR = "/mnt/hpfs/HDU/ssk/SA-1B_256"
TARGET_SIZE = 256


def resize_one(filename, src_dir, dst_dir):
    """缩放单张图片，返回 (成功, 文件名)"""
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)
    if os.path.exists(dst_path):
        return (True, filename)
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            img.save(dst_path, "JPEG", quality=95)
        return (True, filename)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return (False, filename)


def rewrite_manifest(src_manifest, dst_manifest, src_dir, dst_dir):
    """重写 manifest，将路径从 sa1b_1024 替换为 sa1b_256"""
    count = 0
    with open(src_manifest, "r") as fin, open(dst_manifest, "w") as fout:
        for line in fin:
            entry = json.loads(line.strip())
            old_path = entry["image_path"]
            new_path = old_path.replace(src_dir, dst_dir)
            fout.write(json.dumps({"image_path": new_path}) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(16, cpu_count()),
                        help="并行 worker 数量")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(DST_DIR, exist_ok=True)
    print(f"源目录: {SRC_DIR}")
    print(f"目标目录: {DST_DIR}")
    print(f"目标分辨率: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Workers: {args.workers}")

    # 获取所有图片文件名
    filenames = [f for f in os.listdir(SRC_DIR) if f.endswith(".jpg")]
    filenames.sort()
    total = len(filenames)
    print(f"共 {total} 张图片待处理")

    # 检查已完成数量
    existing = set(os.listdir(DST_DIR))
    remaining = [f for f in filenames if f not in existing]
    print(f"已完成: {total - len(remaining)}, 剩余: {len(remaining)}")

    if remaining:
        # 多进程缩放
        worker_fn = partial(resize_one, src_dir=SRC_DIR, dst_dir=DST_DIR)
        done = 0
        failed = 0
        with Pool(args.workers) as pool:
            for success, fname in pool.imap_unordered(worker_fn, remaining, chunksize=256):
                if success:
                    done += 1
                else:
                    failed += 1
                if (done + failed) % 10000 == 0:
                    print(f"  进度: {done + failed}/{len(remaining)}  "
                          f"(成功: {done}, 失败: {failed})")

        print(f"\n缩放完成: 成功 {done}, 失败 {failed}")
    else:
        print("所有图片已缩放，跳过")

    # 生成新的 manifest
    print("\n生成 manifest 文件...")
    for name in ["train_manifest.jsonl", "val_manifest.jsonl"]:
        src_m = os.path.join(SRC_MANIFEST_DIR, name)
        dst_m = os.path.join(DST_MANIFEST_DIR, name)
        if os.path.exists(src_m):
            n = rewrite_manifest(src_m, dst_m, SRC_DIR, DST_DIR)
            print(f"  {name}: {n} 条")
        else:
            print(f"  {name}: 源文件不存在，跳过")

    print("\n全部完成!")
    print(f"  图片目录: {DST_DIR}")
    print(f"  训练 manifest: {DST_MANIFEST_DIR}/train_manifest.jsonl")
    print(f"  验证 manifest: {DST_MANIFEST_DIR}/val_manifest.jsonl")


if __name__ == "__main__":
    main()

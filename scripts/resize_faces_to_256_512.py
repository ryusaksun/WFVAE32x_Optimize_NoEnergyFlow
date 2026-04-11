"""
将人脸数据集 (images1024x1024) 缩放到 256px 和 512px，切分训练/验证集，
并追加到现有 SA-1B 的 manifest 中。

用法:
    python resize_faces_to_256_512.py [--workers 16] [--train_ratio 0.9] [--seed 42]

输入: /mnt/hpfs/images1024x1024/*.png (1024px)
输出:
    /mnt/hpfs/HDU/ssk/SA-1B_256/faces_256/*.jpg
    /mnt/hpfs/HDU/ssk/SA-1B_512/faces_512/*.jpg
    追加到 SA-1B_256 和 SA-1B_512 的 train/val manifest
"""

import os
import json
import random
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from PIL import Image

SRC_DIR = "/mnt/hpfs/images1024x1024"
DST_DIR_256 = "/mnt/hpfs/HDU/ssk/SA-1B_256/faces_256"
DST_DIR_512 = "/mnt/hpfs/HDU/ssk/SA-1B_512/faces_512"

MANIFEST_256_DIR = "/mnt/hpfs/HDU/ssk/SA-1B_256"
MANIFEST_512_DIR = "/mnt/hpfs/HDU/ssk/SA-1B_512"


def resize_one(filename, src_dir, dst_dir_256, dst_dir_512):
    """缩放单张图片到 256 和 512，返回 (成功, 文件名)"""
    src_path = os.path.join(src_dir, filename)
    dst_name = os.path.splitext(filename)[0] + ".jpg"
    dst_path_256 = os.path.join(dst_dir_256, dst_name)
    dst_path_512 = os.path.join(dst_dir_512, dst_name)

    done_256 = os.path.exists(dst_path_256)
    done_512 = os.path.exists(dst_path_512)
    if done_256 and done_512:
        return (True, filename)

    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            if not done_512:
                img_512 = img.resize((512, 512), Image.LANCZOS)
                img_512.save(dst_path_512, "JPEG", quality=95)
            if not done_256:
                img_256 = img.resize((256, 256), Image.LANCZOS)
                img_256.save(dst_path_256, "JPEG", quality=95)
        return (True, filename)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return (False, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(16, cpu_count()))
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(DST_DIR_256, exist_ok=True)
    os.makedirs(DST_DIR_512, exist_ok=True)

    # 获取所有 PNG 图片
    filenames = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".png"))
    total = len(filenames)
    print(f"源目录: {SRC_DIR}")
    print(f"共 {total} 张图片")
    print(f"Workers: {args.workers}")

    # 多进程缩放
    worker_fn = partial(resize_one, src_dir=SRC_DIR,
                        dst_dir_256=DST_DIR_256, dst_dir_512=DST_DIR_512)
    done = 0
    failed = 0
    with Pool(args.workers) as pool:
        for success, fname in pool.imap_unordered(worker_fn, filenames, chunksize=256):
            if success:
                done += 1
            else:
                failed += 1
            if (done + failed) % 5000 == 0:
                print(f"  进度: {done + failed}/{total}  "
                      f"(成功: {done}, 失败: {failed})")

    print(f"\n缩放完成: 成功 {done}, 失败 {failed}")

    # 切分训练/验证集（固定 seed）
    random.seed(args.seed)
    shuffled = filenames[:]
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * args.train_ratio)
    train_files = sorted(shuffled[:split_idx])
    val_files = sorted(shuffled[split_idx:])
    print(f"\n切分: 训练集 {len(train_files)} 张, 验证集 {len(val_files)} 张 "
          f"(ratio={args.train_ratio}, seed={args.seed})")

    # 追加到 manifest
    print("\n追加 manifest...")
    for split_name, split_files in [("train_manifest.jsonl", train_files),
                                     ("val_manifest.jsonl", val_files)]:
        for res, dst_dir, manifest_dir in [
            (256, DST_DIR_256, MANIFEST_256_DIR),
            (512, DST_DIR_512, MANIFEST_512_DIR),
        ]:
            manifest_path = os.path.join(manifest_dir, split_name)
            count = 0
            with open(manifest_path, "a") as f:
                for filename in split_files:
                    jpg_name = os.path.splitext(filename)[0] + ".jpg"
                    img_path = os.path.join(dst_dir, jpg_name)
                    f.write(json.dumps({"image_path": img_path}) + "\n")
                    count += 1
            print(f"  {manifest_path}: +{count} 条")

    print("\n全部完成!")


if __name__ == "__main__":
    main()

import argparse

import numpy as np
import torch
from einops import rearrange
from torchvision import transforms
from PIL import Image

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from causalimagevae.model import *


def preprocess(image_data: Image.Image, short_size: int = 256, keep_ratio: bool = False) -> torch.Tensor:
    if keep_ratio:
        # Keep original aspect ratio, only pad to multiple of 8
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        outputs = transform(image_data).unsqueeze(0)
        _, _, H, W = outputs.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            outputs = torch.nn.functional.pad(outputs, (0, pad_w, 0, pad_h), mode='reflect')
        return outputs, H, W
    else:
        # Original behavior: Resize -> CenterCrop -> square
        transform = transforms.Compose(
            [
                transforms.Resize(short_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(short_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        outputs = transform(image_data)
        outputs = outputs.unsqueeze(0)
        return outputs, None, None


def main(args: argparse.Namespace):
    device = args.device
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    data_type = dtype_map[args.dtype]

    model_cls = ModelRegistry.get_model(args.model_name)
    vae = model_cls.from_pretrained(args.from_pretrained)
    vae = vae.to(device, dtype=data_type)
    vae.eval()

    with torch.no_grad():
        x_vae, orig_h, orig_w = preprocess(
            Image.open(args.image_path).convert("RGB"),
            args.short_size,
            keep_ratio=args.keep_ratio,
        )
        x_vae = x_vae.to(device, dtype=data_type)

        latents = vae.encode(x_vae).latent_dist.sample()
        latents = latents.to(data_type)
        image_recon = vae.decode(latents).sample

    # Crop back to original size if padded
    if orig_h is not None:
        image_recon = image_recon[:, :, :orig_h, :orig_w]

    x = image_recon[0]  # (C, H, W)
    x = x.detach().float().cpu().numpy()
    x = np.clip(x, -1, 1)
    x = (x + 1) / 2
    x = (255 * x).astype(np.uint8)
    x = rearrange(x, "c h w -> h w c")
    image = Image.fromarray(x)
    image.save(args.rec_path)
    print(f"Reconstructed image saved to {args.rec_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image file"
    )
    parser.add_argument(
        "--rec_path", type=str, required=True, help="Path to save the reconstructed image"
    )
    parser.add_argument(
        "--model_name", type=str, default="WFIVAE2", help="Name of the model to use"
    )
    parser.add_argument(
        "--from_pretrained",
        type=str,
        required=True,
        help="Path or identifier of the pretrained model",
    )
    parser.add_argument(
        "--short_size", type=int, default=256, help="Short side size for resizing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation (e.g., 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for inference (default: bf16)",
    )
    parser.add_argument(
        "--keep_ratio",
        action="store_true",
        help="Keep original aspect ratio instead of center-cropping to square",
    )

    args = parser.parse_args()
    main(args)

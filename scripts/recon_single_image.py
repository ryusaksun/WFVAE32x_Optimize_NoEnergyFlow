import argparse

import numpy as np
import torch
from einops import rearrange
from torchvision import transforms
from PIL import Image

import sys

sys.path.append(".")
from causalimagevae.model import *


def preprocess(image_data: Image.Image, short_size: int = 256) -> torch.Tensor:
    # Match training preprocessing: Resize -> CenterCrop -> ToTensor -> Normalize[-1,1]
    transform = transforms.Compose(
        [
            transforms.Resize(short_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(short_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    outputs = transform(image_data)
    outputs = outputs.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    return outputs


def main(args: argparse.Namespace):
    device = args.device
    data_type = torch.bfloat16

    model_cls = ModelRegistry.get_model(args.model_name)
    vae = model_cls.from_pretrained(args.from_pretrained)
    vae = vae.to(device, dtype=data_type)
    vae.eval()

    with torch.no_grad():
        x_vae = preprocess(Image.open(args.image_path).convert("RGB"), args.short_size)
        x_vae = x_vae.to(device, dtype=data_type)

        latents = vae.encode(x_vae).latent_dist.sample()
        latents = latents.to(data_type)
        image_recon = vae.decode(latents).sample

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

    args = parser.parse_args()
    main(args)

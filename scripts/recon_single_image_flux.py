import argparse
import numpy as np
import torch
from einops import rearrange
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKLFlux2


def preprocess(image_data: Image.Image, short_size: int = 512, keep_ratio: bool = False) -> torch.Tensor:
    if keep_ratio:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        outputs = transform(image_data).unsqueeze(0)
        _, _, H, W = outputs.shape
        # FLUX.2 VAE: 4 DownEncoderBlock2D (each 2x) + patch_size 2 => need multiple of 32
        align = 32
        pad_h = (align - H % align) % align
        pad_w = (align - W % align) % align
        if pad_h > 0 or pad_w > 0:
            outputs = torch.nn.functional.pad(outputs, (0, pad_w, 0, pad_h), mode='reflect')
        return outputs, H, W
    else:
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


def compute_image_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    orig = (original.float() + 1) / 2
    recon = (reconstructed.float() + 1) / 2
    mse = torch.mean((orig - recon) ** 2)
    if mse == 0:
        return 100.0
    return (20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))).item()


def main(args: argparse.Namespace):
    device = args.device
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    data_type = dtype_map[args.dtype]

    print(f"Loading FLUX.2 VAE from: {args.model_path}")
    vae = AutoencoderKLFlux2.from_pretrained(args.model_path, torch_dtype=data_type)
    vae = vae.to(device)
    vae.eval()

    print(f"  latent_channels: {vae.config.latent_channels}")
    if hasattr(vae.config, 'scaling_factor'):
        print(f"  scaling_factor: {vae.config.scaling_factor}")
    if hasattr(vae.config, 'shift_factor'):
        print(f"  shift_factor: {vae.config.shift_factor}")

    with torch.no_grad():
        x_vae, orig_h, orig_w = preprocess(
            Image.open(args.image_path).convert("RGB"),
            args.short_size,
            keep_ratio=args.keep_ratio,
        )
        x_vae = x_vae.to(device, dtype=data_type)
        print(f"  Input shape: {x_vae.shape}")

        latents = vae.encode(x_vae).latent_dist.mode()
        print(f"  Latent shape: {latents.shape}")

        image_recon = vae.decode(latents, return_dict=False)[0]

    if orig_h is not None:
        image_recon = image_recon[:, :, :orig_h, :orig_w]

    if args.compute_metrics:
        if orig_h is not None:
            x_orig = x_vae[:, :, :orig_h, :orig_w]
        else:
            x_orig = x_vae
        psnr = compute_image_psnr(x_orig[0].cpu(), image_recon[0].cpu())
        print(f"  PSNR: {psnr:.2f} dB")

    x = image_recon[0]
    x = x.detach().float().cpu().numpy()
    x = np.clip(x, -1, 1)
    x = (x + 1) / 2
    x = (255 * x).astype(np.uint8)
    x = rearrange(x, "c h w -> h w c")
    image = Image.fromarray(x)
    image.save(args.rec_path)
    print(f"Reconstructed image saved to {args.rec_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX.2 VAE image reconstruction")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--rec_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Local path to FLUX.2 VAE directory")
    parser.add_argument("--short_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--keep_ratio", action="store_true")
    parser.add_argument("--compute_metrics", action="store_true")
    args = parser.parse_args()
    main(args)

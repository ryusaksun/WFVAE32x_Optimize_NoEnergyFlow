import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class ManifestImageDataset(Dataset):
    """
    Dataset for loading images from a JSONL manifest file.

    Manifest format (JSONL):
    {"image_path": "relative/path/to/image.jpg", "caption": "optional", "metadata": {...}}
    {"image_path": "relative/path/to/image.png", "caption": "optional", "metadata": {...}}
    """

    def __init__(
        self,
        manifest_path,
        base_dir=None,
        resolution=256,
        transform=None,
    ):
        """
        Args:
            manifest_path: Path to JSONL manifest file
            base_dir: Base directory for image paths (if paths in manifest are relative)
            resolution: Target resolution for images
            transform: Optional custom transform
        """
        self.manifest_path = manifest_path
        self.base_dir = base_dir or os.path.dirname(manifest_path)
        self.resolution = resolution

        # Load manifest
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples from {manifest_path}")

        # Default transform
        self.transform = transform or T.Compose([
            T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, _retry=0):
        sample = self.samples[idx]

        # Get image path - support multiple field names
        image_path = sample.get('image_path', sample.get('path', sample.get('target', '')))
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.base_dir, image_path)

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)

            # Return dict with image and metadata
            return {
                "image": image,
                "label": sample.get('caption', ''),
                "path": image_path,
                "metadata": sample.get('metadata', {}),
            }
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            if _retry >= 10:
                raise RuntimeError(f"Failed to load image after {_retry} retries, last path: {image_path}")
            return self.__getitem__((idx + 1) % len(self), _retry=_retry + 1)


class ValidManifestImageDataset(Dataset):
    """Validation dataset for images from manifest."""

    def __init__(
        self,
        manifest_path,
        base_dir=None,
        resolution=256,
        crop_size=None,
        transform=None,
    ):
        self.manifest_path = manifest_path
        self.base_dir = base_dir or os.path.dirname(manifest_path)
        self.resolution = resolution

        # Load manifest
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} validation samples from {manifest_path}")

        # Transform
        if crop_size is not None:
            self.transform = transform or T.Compose([
                T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(crop_size),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform or T.Compose([
                T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(resolution),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, _retry=0):
        sample = self.samples[idx]

        # Get image path - support multiple field names
        image_path = sample.get('image_path', sample.get('path', sample.get('target', '')))
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.base_dir, image_path)

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            file_name = os.path.basename(image_path)

            return {
                "image": image,
                "file_name": file_name,
                "index": idx,
                "metadata": sample.get('metadata', {}),
            }
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            if _retry >= 10:
                raise RuntimeError(f"Failed to load validation image after {_retry} retries, last path: {image_path}")
            next_idx = (idx + 1) % len(self) if len(self) > 1 else 0
            return self.__getitem__(next_idx, _retry=_retry + 1)

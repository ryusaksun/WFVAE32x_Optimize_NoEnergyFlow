import os.path as osp
import random
from glob import glob
from torchvision import transforms
import torch
import torch.utils.data as data
from PIL import Image
import pickle


class ImageDataset(data.Dataset):
    """Dataset for loading images for VAE training."""

    image_exts = ["jpg", "jpeg", "png", "webp", "bmp"]

    def __init__(
        self,
        image_folder,
        resolution=256,
        cache_file="image_cache.pkl",
        is_main_process=False,
    ):
        self.resolution = resolution
        self.image_folder = image_folder
        self.cache_file = cache_file
        self.is_main_process = is_main_process

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
        ])

        print("Building image dataset...")
        self.samples = self._make_dataset()
        print(f"Found {len(self.samples)} images")

    def _make_dataset(self):
        cache_file = osp.join(self.image_folder, self.cache_file)

        if osp.exists(cache_file):
            with open(cache_file, "rb") as f:
                samples = pickle.load(f)
        else:
            samples = []
            for ext in self.image_exts:
                samples += glob(osp.join(self.image_folder, "**", f"*.{ext}"), recursive=True)
                samples += glob(osp.join(self.image_folder, "**", f"*.{ext.upper()}"), recursive=True)

            if self.is_main_process and len(samples) > 0:
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump(samples, f)
                except Exception as e:
                    print(f"Error saving cache: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.samples[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            return dict(image=image, label="", path=image_path)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))


class ValidImageDataset(data.Dataset):
    """Validation dataset for images."""

    image_exts = ["jpg", "jpeg", "png", "webp", "bmp"]

    def __init__(
        self,
        image_folder,
        resolution=256,
        crop_size=None,
        cache_file="valid_image_cache.pkl",
        is_main_process=False,
    ):
        self.resolution = resolution
        self.image_folder = image_folder
        self.is_main_process = is_main_process

        if crop_size is not None:
            self.transform = transforms.Compose([
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        self.samples = self._make_dataset(cache_file)
        print(f"Found {len(self.samples)} validation images")

    def _make_dataset(self, cache_file):
        cache_path = osp.join(self.image_folder, cache_file)

        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                samples = pickle.load(f)
        else:
            samples = []
            for ext in self.image_exts:
                samples += glob(osp.join(self.image_folder, "**", f"*.{ext}"), recursive=True)
                samples += glob(osp.join(self.image_folder, "**", f"*.{ext.upper()}"), recursive=True)

            if self.is_main_process and len(samples) > 0:
                try:
                    with open(cache_path, "wb") as f:
                        pickle.dump(samples, f)
                except Exception as e:
                    print(f"Error saving cache: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.samples[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            file_name = osp.basename(image_path)
            return dict(image=image, file_name=file_name, index=idx)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return self.__getitem__(0)

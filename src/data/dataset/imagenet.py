import os
import torch
import torchvision.transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Normalize
from functools import partial

import numpy as np

def center_crop_fn(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class LocalCachedDataset(ImageFolder):
    def __init__(self, root, resolution=256, cache_root=None):
        super().__init__(root)
        self.cache_root = cache_root
        self.transform = partial(center_crop_fn, image_size=resolution)

    def load_latent(self, latent_path):
        pk_data = torch.load(latent_path)
        mean = pk_data['mean'].to(torch.float32)
        logvar = pk_data['logvar'].to(torch.float32)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        latent = mean + torch.randn_like(mean) * std
        return latent

    def __getitem__(self, idx: int):
        image_path, target = self.samples[idx]
        latent_path = image_path.replace(self.root, self.cache_root) + ".pt"

        raw_image = Image.open(image_path).convert('RGB')
        raw_image = self.transform(raw_image)
        raw_image = to_tensor(raw_image)
        if self.cache_root is not None:
            latent = self.load_latent(latent_path)
        else:
            latent = raw_image

        metadata = {
            "raw_image": raw_image,
            "class": target,
        }
        return latent, target, metadata


class PixImageNet(ImageFolder):
    def __init__(self, root, resolution=256, random_crop=False, random_flip=False, cache_path=None):
        super().__init__(root)
        self.cache_path = cache_path
        if random_crop:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(resolution),
                    torchvision.transforms.RandomCrop(resolution),
                    torchvision.transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            if random_flip is False:
                self.transform = partial(center_crop_fn, image_size=resolution)
            else:
                self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.Lambda(partial(center_crop_fn, image_size=resolution)),
                    torchvision.transforms.RandomHorizontalFlip(),
                ])
            
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def load_latent(self, latent_path):
        data = np.load(latent_path)
        if len(data.shape) == 3:
            # assume c,h,w
            return torch.from_numpy(data)
        elif len(data.shape) == 4:
            # assume 1, c, h, w
            return torch.from_numpy(data[0])
        else:
             raise ValueError(f"Unknown shape {data.shape} for {latent_path}")

    def __getitem__(self, idx: int):
        image_path, target = self.samples[idx]
        raw_image = Image.open(image_path).convert('RGB')
        raw_image = self.transform(raw_image)
        raw_image = to_tensor(raw_image)

        if self.cache_path is not None:
             # assume structure is identical
             # filename mapping:  image.jpg -> img-mean-std-image.npy
             # or just image.npy?
             # User ls: img-mean-std-000000000.npy
             # Assuming image is 000000000.jpg
             rel_path = os.path.relpath(image_path, self.root)
             folder, filename = os.path.split(rel_path)
             filename_no_ext = os.path.splitext(filename)[0]
             
             latent_filename = f"img-mean-std-{filename_no_ext}.npy"
             latent_path = os.path.join(self.cache_path, folder, latent_filename)
             
             # Fallback if simple mapping doesn't work (e.g. if filenames differ in other ways)
             if not os.path.exists(latent_path):
                  # try direct replace extension
                  latent_path = os.path.join(self.cache_path, folder, filename_no_ext + ".npy")
             
             if os.path.exists(latent_path):
                 latent_data = self.load_latent(latent_path)
                 # sample if mean/std?
                 # content is [mean, std] concatenated in channel dim?
                 if latent_data.shape[0] == 8: # 2*4 channels
                     mean, std = latent_data.chunk(2, dim=0)
                     latent_data = mean + torch.randn_like(mean) * std
                 
                 normalized_image = latent_data
             else:
                 # fallback to encode on fly if not found? 
                 # But model expects latent.
                 print(f"Warning: Latent not found at {latent_path}, using raw image")
                 normalized_image = self.normalize(raw_image)
        else:
             normalized_image = self.normalize(raw_image)

        metadata = {
            "raw_image": raw_image,
            "class": target,
        }
        return normalized_image, target, metadata
import os
import io
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class AddGaussianNoise:
    """
    Adds Gaussian noise to a normalised tensor.
    Simulates sensor noise and low-quality captures.
    """
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class JPEGCompression:
    """
    Simulates JPEG compression artifacts via PIL encode/decode round-trip.
    """
    def __init__(self, quality_range=(50, 95)):
        self.quality_range = quality_range

    def __call__(self, img):
        quality = int(np.random.randint(self.quality_range[0], self.quality_range[1]))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).copy()


class DeepfakeDataset(Dataset):
    """
    Dataset loader for binary deepfake classification.
    label 0 = real, label 1 = fake.
    """
    def __init__(self, real_dir, fake_dir, train=True):
        self.data = []
        valid_ext = ('.jpg', '.jpeg', '.png')

        for fname in sorted(os.listdir(real_dir)):
            if fname.lower().endswith(valid_ext):
                self.data.append((os.path.join(real_dir, fname), 0))

        for fname in sorted(os.listdir(fake_dir)):
            if fname.lower().endswith(valid_ext):
                self.data.append((os.path.join(fake_dir, fname), 1))

        self.num_real = sum(1 for _, l in self.data if l == 0)
        self.num_fake = sum(1 for _, l in self.data if l == 1)

        print(f"[Dataset] Loaded {self.num_real} real | {self.num_fake} fake "
              f"| mode={'train' if train else 'val'}")

        norm_mean = [0.485, 0.456, 0.406]
        norm_std  = [0.229, 0.224, 0.225]

        if train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                # --- Geometric ---
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                # --- Photometric (Probabilities kept at 0.2 to prevent underfitting) ---
                transforms.RandomApply([JPEGCompression(quality_range=(50, 95))], p=0.2),
                transforms.ToTensor(),
                transforms.RandomApply([AddGaussianNoise(std=0.02)], p=0.2),
                transforms.Normalize(norm_mean, norm_std),
                # --- Coarse dropout ---
                transforms.RandomErasing(
                    p=0.2,
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3),
                    value=0,
                    inplace=False,
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ])

    def get_sample_weights(self):
        """
        Returns per-sample weights for WeightedRandomSampler.
        """
        weight_per_class = {
            0: 1.0 / self.num_real,
            1: 1.0 / self.num_fake,
        }
        return [weight_per_class[label] for _, label in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not read image: {img_path} — skipping with zeros")
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)
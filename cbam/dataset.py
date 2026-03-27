import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, train=True):
        self.data = []
        valid_ext = ('.jpg', '.jpeg', '.png')
        for img in os.listdir(real_dir):
            if img.lower().endswith(valid_ext):
                self.data.append((os.path.join(real_dir, img), 0))
        for img in os.listdir(fake_dir):
            if img.lower().endswith(valid_ext):
                self.data.append((os.path.join(fake_dir, img), 1))

        # Standard ImageNet Normalization
        norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        if train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)
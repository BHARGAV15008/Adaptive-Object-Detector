import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),  # Resize images to YOLO input size
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        # Extract label from the file name
        label = self.image_files[idx].split('_')[0]  # Assuming label is the prefix before '_'
        label = int(label)  # Convert label to integer

        return image, label

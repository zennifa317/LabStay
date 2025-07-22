import os

from PIL import Image
from sklearn.preprocessing import LabelBinarizer
import torch
from torch.utils.data import Dataset

class LabStayDataset(Dataset):
    def __init__(self, image_paths, label_paths, classes, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths

        self.lb = LabelBinarizer()
        self.lb.fit(classes)
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        
        image = self.load_image(image_path)
        label = self.lb.transform([self.label_paths[image_name]])
        label = torch.tensor(label, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def load_image(self, image_path):
        return Image.open(image_path).convert('RGB')
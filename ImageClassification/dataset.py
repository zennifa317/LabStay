import json
import os

from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset

class LabStayDataset(Dataset):
    def __init__(self, image_paths, label_paths, classes, transform=None):
        with open(image_paths, 'r') as f:
            self.image_paths = f.read().splitlines()
            
        with open(label_paths, 'r') as f:
            self.label_paths = json.load(f)['Annotations']
        
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
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def load_image(self, image_path):
        return Image.open(image_path).convert('RGB')
import os
import json

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset import LabStayDataset
from model import LabStayModel

def eval(model, dataloader, device):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for images, label in dataloader:
            images = images.to(device)
            pred = model(images)
            
            preds.extend(pred.argmax(dim=1).cpu().numpy())
            labels.extend(label.squeeze().argmax(dim=1).cpu().numpy())

    accuracy = accuracy_score(np.array(labels), np.array(preds))

    return accuracy

if __name__ == '__main__':
    # 必須
    weight_path = 'path/to/weight/best.pth'
    data_path = './data/data.json'
    label_path = 'path/to/annotations.json'
    test_batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #output_dir = './results/test'

    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)

    print(f'Using device: {device}')
    
    with open(data_path, 'r') as f:
        data = json.load(f)

    test_images = data['test']
    classes = data['class']
    num_classes = len(classes)

    with open(test_images, 'r') as f:
        test_images = f.read().splitlines()

    with open(label_path, 'r') as f:
        label_paths = json.load(f)['annotations']

    model = LabStayModel()
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.to(device)
    
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = LabStayDataset(test_images, label_paths, classes, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    accuracy = eval(model, test_dataloader, device)
    print(f'Test Accuracy: {accuracy:.4f}')
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LabStayDataset
from model import LabStayModel

if __name__ == '__main__':
    data_path = './data/data.json'
    label_path = './data/labels.json'
    train_batch_size = 32
    valid_batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    
    print(f'Using device: {device}')
    
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    train_images = data['train']
    val_images = data['val']
    classes = data['class']
    
    with open(label_path, 'r') as f:
        label_paths = json.load(f)['Annotations']
    
    train_dataset = LabStayDataset(train_images, label_paths, classes)
    valid_dataset = LabStayDataset(val_images, label_paths, classes)
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)
    
    model = LabStayModel()
    model.to(device)
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    
    for epoch in tqdm(range(epochs)):
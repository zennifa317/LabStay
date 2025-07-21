import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from dataset import LabStayDataset
from model import LabStayModel

def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.squeeze().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    ave_loss = total_loss / len(dataloader)

    return ave_loss

def valid(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.squeeze().to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
    
    ave_loss = total_loss / len(dataloader)

    return ave_loss

def plot_loss(train_losses, valid_losses, epochs, results_path):
    plt.figure(figsize=(10, 5))
    epochs_list = list(range(1, epochs + 1))
    plt.plot(epochs_list, train_losses, label='Train Loss')
    plt.plot(epochs_list, valid_losses, label='Valid Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()

    plt.show()

    plot_path = os.path.join(results_path, 'loss_plot.png')
    plt.savefig(plot_path)

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
    output_dir = './results/train'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f'Using device: {device}')
    
    with open(data_path, 'r') as f:
        data = json.load(f)

    train_images = data['train']
    val_images = data['val']
    classes = data['class']

    with open(train_images, 'r') as f:
        train_images = f.read().splitlines()

    with open(val_images, 'r') as f:
        val_images = f.read().splitlines()

    with open(label_path, 'r') as f:
        label_paths = json.load(f)['Annotations']
        
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = LabStayDataset(train_images, label_paths, classes, transform=train_transform)
    valid_dataset = LabStayDataset(val_images, label_paths, classes, transform=valid_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

    model = LabStayModel()
    model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

    best_loss = float('inf')

    best_weight_path = os.path.join(output_dir, 'best.pth')
    last_weight_path = os.path.join(output_dir, 'last.pth')

    train_losses = []
    valid_losses = []

    print('Starting training...')
    for epoch in tqdm(range(epochs)):
        train_loss = train(train_dataloader, model, loss, optimizer, device)
        valid_loss = valid(valid_dataloader, model, loss, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), best_weight_path)
            
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
        
        scheduler.step()

    plot_loss(train_losses, valid_losses, epochs, output_dir)

    print('Training complete.')
    
    torch.save(model.state_dict(), last_weight_path)
    print(f'Last Epoch Model saved at {last_weight_path}')
    print(f'Best Model saved at {best_weight_path}')
from PIL import Image
import torch
from torchvision.transforms import v2

from model import LabStayModel

def detect(image_path, model, device):
    model.eval()

    detect_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        image = detect_transform(image)
        image = image.unsqueeze(0).to(device)
        
        pred = model(image)
        pred = pred.item()

    return pred

if __name__ == '__main__':
    device = 'cuda'
    weight_path = 'path/to/weights.pth'
    image_path = 'path/to/image.jpg'
    top_k = 1
    
    model = LabStayModel()
    model.to(device)

    model.load_state_dict(torch.load(weight_path))

    pred = detect(image_path, model, device)
    values, indices = torch.topk(pred, top_k)

    print(f'Top {top_k} predictions:')
    for v, i in zip(values, indices):
        print(f'Class: {i.item()}, Score: {v.item():.4f}')
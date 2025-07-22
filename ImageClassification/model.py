import torch.nn as nn

class LabStayModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(3, 6, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 6, kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 7 * 7, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        
        return x
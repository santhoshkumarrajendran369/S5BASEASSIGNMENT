import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv block with efficient filters
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second conv block with slight increase
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Efficient attention mechanism
        self.se1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 4, 1),
            nn.ReLU(),
            nn.Conv2d(4, 16, 1),
            nn.Sigmoid()
        )
        
        # Third conv block
        self.conv3 = nn.Conv2d(16, 20, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(20)
        
        # Fully connected layers
        self.fc1 = nn.Linear(20 * 3 * 3, 80)
        self.fc2 = nn.Linear(80, 10)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # First block with strong feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.pool(x)
        
        # Second block with attention
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.1)
        
        # Apply squeeze-excitation
        att = self.se1(x)
        x = x * att
        x = self.pool(x)
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.1)
        x = self.pool(x)
        
        # Fully connected with minimal dropout
        x = x.view(-1, 20 * 3 * 3)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout(x)
        x = self.fc2(x)
        return x 
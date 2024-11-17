import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv block with more filters
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second conv block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Third conv block with residual connection
        self.conv3a = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(32)
        self.conv3b = nn.Conv2d(32, 32, kernel_size=1)  # 1x1 conv for channel attention
        self.bn3b = nn.BatchNorm2d(32)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 3 * 3, 64)  # Reduced to stay under parameter limit
        self.fc2 = nn.Linear(64, 10)
        
        # Other layers
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Increased dropout for better regularization
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.pool(x)
        
        # Third block with attention and residual
        identity = x
        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.relu(x)
        attention = self.conv3b(x)
        attention = self.bn3b(attention)
        attention = torch.sigmoid(attention)  # Channel attention
        x = x * attention + identity  # Attention-weighted residual
        x = self.dropout2(x)
        x = self.pool(x)
        
        # Fully connected
        x = x.view(-1, 32 * 3 * 3)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x 
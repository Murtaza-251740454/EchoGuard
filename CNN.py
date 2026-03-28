import torch.nn as nn
import torch.nn.functional as F 

class CNNForMFCC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 10 * 43, 128)  # Adjusted for [40, 174] input
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # [B, 16, 20, 87]
        x = self.pool2(F.relu(self.conv2(x)))  # [B, 32, 10, 43]
        x = self.dropout(x.view(x.size(0), -1))
        x = F.relu(self.fc1(x))
        return self.fc2(x)
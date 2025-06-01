import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):  # 4 classes: covid, normal, lung_opacity, viral_pneumonia
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 28 * 28, 256)  # adjust 28*28 based on input image size
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 1/2 size
        x = self.pool(F.relu(self.conv2(x)))  # 1/4 size
        x = self.pool(F.relu(self.conv3(x)))  # 1/8 size

        x = x.view(-1, 64 * 28 * 28)  # flatten

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

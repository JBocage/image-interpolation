import torch
from models.abstract_model import AbstractModel

import torch.nn as nn

class Encoder(AbstractModel):
    """Used to encode an MNIST image"""
    
    def __init__(self):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1, 28, 28
        print(x.shape)        
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        # 16, 14, 14
        print(x.shape)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.relu(x)
        # 32, 7, 7
        print(x.shape)
        x = self.conv3(x)
        x = self.relu(x)
        # 64, 3, 3
        print(x.shape)
        x = self.conv4(x)
        x = self.relu(x)
        # 64, 1, 1
        print(x.shape)
        x = self.conv5(x)
        x = self.relu(x)
        # 16, 1, 1
        return x

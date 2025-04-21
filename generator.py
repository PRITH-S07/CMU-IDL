import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseGenerator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32,       32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, channels, 3, padding=1)        
        )

    def forward(self, x):
        noise = self.net(x)
        return x + noise, noise
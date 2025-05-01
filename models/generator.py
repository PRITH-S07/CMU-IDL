import torch
import torch.nn as nn


class NoiseGenerator(nn.Module):
    def __init__(self, embedding_dim=768):
        super(NoiseGenerator, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        noise = self.relu(self.fc1(x))
        noise = self.relu(self.fc2(noise))
        noise = self.fc3(noise)
        perturbed_embedding = x + noise
        return perturbed_embedding, noise

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def string_to_np(feature_str):
    embedding_list = np.fromstring(feature_str[1:-1], sep=' ')
    assert len(embedding_list) == 768
    return np.array(embedding_list.astype(np.float32))

class ArtEmbeddingDataset(Dataset):
    def __init__(self, csv_file, transform=None, ai_only=False):
        self.data = pd.read_csv(csv_file, usecols=["Filepath", "Features", "Label"])
        self.transform = transform
        if ai_only:
            self.data = self.data[self.data["Label"] == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filepath = row["Filepath"]
        features = string_to_np(row["Features"])
        label = row["Label"]
        sample = {"filepath": filepath, "features": features, "label": label}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class AITestDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "label": self.labels[idx]
        }
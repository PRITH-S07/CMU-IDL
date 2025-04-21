import os
import ast
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# DIFFUSION_MODELS = ["openjourney", "titan", "dalle", "real"]
# ROOT = '/ocean/projects/cis250019p/kanand/IDL Image Generation'

# def collect_images(generators):
#   generated_images = {}
#   for directory in generators:
#     for filepath in os.listdir(os.path.join(ROOT, "data", directory)):
#       if filepath.lower().endswith(".jpg"):
#         full_path = os.path.join(ROOT, "data", directory, filepath)
#         id_idx = filepath.rfind('_') + 1
#         id = filepath[id_idx:-4]
#         label = 1 if directory == "real" else 0
#         generated_images[full_path] = {
#             "generator": directory,
#             "label": label, # 0 = fake, 1 = real
#             "id": id,
#         }
#   return generated_images


from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd


class ArtImageDataset(Dataset):
    """
    Returns
    -------
    dict
        {
          "image"   : torch.Tensor  (3, H, W)  in [0,1]
          "label"   : int           0=AI, 1=Real
          "filepath": str           original file path (optional for bookkeeping)
        }
    """

    def __init__(self,
                 csv_file: str,
                 transform=None,
                 ai_only: bool = False,
                 img_size: int = 512):
        """
        Parameters
        ----------
        csv_file : str
            Path to the metadata CSV.  Must have "Filepath" and "Label" columns.
        transform : torchvision transform | None
            If None, a default (Resize -> ToTensor) is used.
        ai_only : bool
            Keep only rows with Label==0 (AI‑generated) when True.
        img_size : int
            Used by the default transform; ignored if you supply your own.
        """
        self.meta = pd.read_csv(csv_file, usecols=["Filepath", "Label", "Features"])
        if ai_only:
            self.meta = self.meta[self.meta["Label"] == 0]

        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()   # -> float32 tensor, scaled to [0,1]
        ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row      = self.meta.iloc[idx]
        filepath = row.Filepath
        features = np.fromstring(row.Features[1:-1], sep=' ', dtype=np.float32)
        label    = int(row.Label)      # 0 (AI) / 1 (Real)
        filepath=filepath.split("MyDrive/")[1]
        # print(filepath)
        try:
            img = Image.open(filepath).convert("RGB")
            img = self.transform(img)

            return {"image": img, "label": label, "filepath": filepath, "embeddings": features}
        except:
            print ("skipped")

def collate_with_embeddings(batch):
      embs, imgs, labs = zip(*batch)
      emb_np = np.stack(embs, axis=0).astype(np.float32)
      imgs_t = torch.stack(imgs, dim=0)
      labels = torch.tensor(labs, dtype=torch.long)
      return {"image": imgs_t, "embedding": emb_np, "label": labels}



class TupleDataset(Dataset):
    """Dataset wrapper for tuples of (embedding, image) pairs with labels"""
    
    def __init__(self, data_tuples, labels):
        """
        Parameters
        ----------
        data_tuples : list of tuples
            Each tuple contains (embedding, image)
        labels : list
            Labels corresponding to each tuple
        """
        self.data = data_tuples
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        embedding, image = self.data[idx]
        label = self.labels[idx]
        return embedding, image, label



def train_test_ai_vs_all(dataset, test_size=0.2, random_state=42):
    
    X_ai, y_ai = [], []      # AI‑generated only
    X_real, y_real = [], []  # Real only

    for idx in tqdm(range(len(dataset)), desc="Processing dataset"):
        try:
            item = dataset[idx]
            pair = (item["embeddings"], item["image"])  # tuple
            if item["label"] == 0:
                X_ai.append(pair)
                y_ai.append(0)
            else:
                X_real.append(pair)
                y_real.append(1)
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            continue

    # split AI samples into train / test
    X_ai_train, X_ai_test, y_ai_train, y_ai_test = train_test_split(
        X_ai, y_ai, test_size=test_size, random_state=random_state, shuffle=True
    )

    # build final sets
    X_train = X_ai_train
    y_train = y_ai_train

    X_test = X_ai_test + X_real          
    y_test = y_ai_test + y_real

    return X_train, X_test, y_train, y_test



# csv_meta = "/ocean/projects/cis250019p/kanand/IDL Image Generation/images_hidden_state_embedding.csv"
# dataset  = ArtImageDataset(csv_meta, ai_only=False)  # we need *all* rows
# X_train, X_test, y_train, y_test = train_test_ai_vs_all(dataset)

# print(len(X_train), "AI images in train set")
# print(len(X_test),  "images in test set (AI + Real)")
# print("first X tuple shapes:", X_train[0][0].shape, X_train[0][1].shape)

# if X_train and X_test:
#     train_ds = TupleDataset(X_train, y_train)        # AI‑only   (label==0)
#     test_ds = TupleDataset(X_test, y_test)           # AI + Real
#     train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
#     print("DataLoaders created successfully")
# else:
#     print("Error: Unable to create DataLoaders due to lack of valid samples")


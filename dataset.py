import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class MNistDataSet(Dataset):
    def __init__(self, file_path, device="cpu"):
        self.data = pd.read_csv(file_path)
        self.len = len(self.data)
        self.device = device

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype(np.float).reshape((1, 28, 28)) / 255.0
        label = self.data.iloc[index, 0]

        return (
            torch.tensor(image, dtype=torch.float).to(self.device),
            torch.tensor(label, dtype=torch.long).to(self.device))

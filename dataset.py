import torch
from torch.utils.data import Dataset

class EmptyDataset(Dataset):
    def __init__(self, len) -> None:
        super().__init__()
        self.len = len

    def __getitem__(self, index):
        return torch.Tensor([0])
    def __len__(self):
        return self.len

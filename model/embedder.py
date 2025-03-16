import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

class IdentityEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(num_classes=2048)
        self.last = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.last(x)
        x = F.normalize(x, p=2, dim=1)
        return x

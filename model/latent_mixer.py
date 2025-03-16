import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentMixerSep(nn.Module):
    def __init__(self, size, residual_coefficient=1) -> None:
        super().__init__()
        self.mixer = nn.ModuleList()

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 2
        self.residual_coefficient = residual_coefficient

        for _ in range(self.num_layers):
            self.mixer.append(nn.Sequential(
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
        ))
    
    def forward(self, source_latent, target_latent):
        _source_latent = source_latent.permute(1,0,2)
        _target_latent = target_latent.permute(1,0,2)

        mix_latent = torch.Tensor().to(source_latent.device)
        for i in range(self.num_layers):
            _source = _source_latent[i]
            _target = _target_latent[i]
            _mix = self.mixer[i](torch.cat((_source, _target), dim=1))

            mix_latent = torch.cat((mix_latent, _mix.unsqueeze(1)), dim=1)
        
        return mix_latent + self.residual_coefficient * target_latent

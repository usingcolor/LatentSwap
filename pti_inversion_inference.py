import os
import math
import argparse
import pickle as pkl
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from torchvision.transforms import transforms

import torch
import torchvision

from latentswap_pl import LatentSwap
from training.coaches.multi_id_coach import MultiIDCoach
from torch.utils.data import DataLoader
from utils.images_dataset import ImagesDataset



parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=int, default=-1)

parser.add_argument('--model_config', type=str, default='config/model.yaml')

parser.add_argument('--model_checkpoint_path', type=str, required=True)

parser.add_argument('--target_image_path', type=str, default=None)
parser.add_argument('--source_image_path', type=str, default=None)

parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--save_path', type=str, default='swap.png')

parser.add_argument('--save_latent', action='store_true')
args = parser.parse_args()



device = torch.device('cpu') if args.gpus == -1 else torch.device(f'cuda:{args.gpus}')

net = LatentSwap(OmegaConf.load(args.model_config))

# PTI inversion
dataset = ImagesDataset([args.source_image_path, args.target_image_path], transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.Resize((1024, 1024)),
    ]))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
coach = MultiIDCoach(dataloader)
style_gan, embedding_list = coach.train()
style_gan.eval()
style_gan.to(device)


img_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((1024, 1024)),
    ])

image_list = []
image_list.append(img_trans(Image.open(args.source_image_path)))
image_list.append(img_trans(Image.open(args.target_image_path)))
checkpoint = torch.load(args.model_checkpoint_path, map_location='cpu')
identity_mixer = net.identity_mixer
new_checkpoint = {}
for key, val in checkpoint['state_dict'].items():
    if 'identity_mixer' in key:
        new_checkpoint[key[15:]] = val
identity_mixer.load_state_dict(new_checkpoint)

identity_mixer.eval()
identity_mixer = net.identity_mixer.to(device)
    
with torch.no_grad():
    latent_mix = identity_mixer(embedding_list[0].to(device), embedding_list[1].to(device))
    sample_mix = style_gan.synthesis(latent_mix, noise_mode='const', force_fp32 = True)
    sample_mix = (1 + sample_mix) /2
    sample_mix = sample_mix.cpu().squeeze()
    image_list.append(sample_mix)


grid_image = torchvision.utils.make_grid(image_list, nrow=9)
grid_image = torchvision.transforms.ToPILImage()(grid_image.squeeze().cpu().clamp(0,1))
grid_image.save(args.save_path)
import random
import pickle as pkl

import wandb
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision

# from model.stylegan2 import Generator
from model.embedder import IdentityEmbedder
from model.latent_mixer import LatentMixerSep
from model.loss import LatentSwapLoss

from dataset import EmptyDataset

from utils.data_utils import instantiate_from_config

class LatentSwap(pl.LightningModule):
    def __init__(self, hp) -> None:
        super().__init__()
        self.hp = hp
        self.latent_type = hp.latent_type
        
        with open(hp.style_gan.checkpoint_path, 'rb') as f:
            old_G = pkl.load(f)['G_ema'].eval()
            old_G = old_G.float()
        self.style_gan = old_G
     
        self.identity_embedder = IdentityEmbedder()
        checkpoint = torch.load(hp.embedder.checkpoint_path, map_location='cpu')    
        new_checkpoint = {}
        for key, val in checkpoint['state_dict'].items():
            if key.split(".")[0] == 'net':
                new_key = key[4:]
                new_checkpoint[new_key] = val
        self.identity_embedder.load_state_dict(new_checkpoint)
        self.identity_embedder.eval()

        if hp.latent_type == 'z' or hp.latent_type == 'w':
            self.identity_mixer = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.SiLU(),
                nn.Linear(1024, 1024),
                nn.SiLU(),
                nn.Linear(1024, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
            )
        elif hp.latent_type == 'wp':
            if 'identity_mixer' in hp:
                residual_coefficient = hp.identity_mixer.residual_coefficient
            else:
                residual_coefficient = 1
            self.identity_mixer = LatentMixerSep(hp.style_gan.args.size, residual_coefficient=residual_coefficient)

        self.style_gan_latent = 512
        self.B = hp.style_gan.batch_size

        self.g_loss = LatentSwapLoss(**hp.loss)

        with open('validation_latent.pkl', 'rb') as f:
            validation_latent = pkl.load(f)
        self.validation_target = validation_latent[0]
        self.validation_source = validation_latent[1]

    def logging_dict(self, log_dict, prefix=None):
        for key, val in log_dict.items():
            if prefix is not None:
                key = prefix + key
            self.log(key, val)

    def logging_image_dict(self, image_dict, prefix=None, commit=False):
        for key, val in image_dict.items():
            if prefix is not None:
                key = prefix + key
            self.logger.experiment.log({key: wandb.Image(val.clamp(0, 1))}, commit=commit)

    def logging_lr(self):
        opts = self.trainer.optimizers
        for idx, opt in enumerate(opts):
            lr = None
            for param_group in opt.param_groups:
                lr = param_group['lr']
                break
            self.log(f"lr_{idx}", lr)
    
    def denorm(self, image):
        return (1 + image) /2

    def common_step(self, batch_idx=None):
        if self.training:
            style_source = torch.randn(self.B, self.style_gan_latent, device=self.device)
            style_target = torch.randn(self.B, self.style_gan_latent, device=self.device)
            
            same = (torch.rand(self.B) > 0.8).view(self.B, 1).to(self.device)
            style_target = same * style_source + (~same) * style_target
        else:
            style_source = self.validation_source[batch_idx].unsqueeze(0).to(self.device)
            style_target = self.validation_target[batch_idx].unsqueeze(0).to(self.device)
            
        # region no_grad
        with torch.no_grad():
            latent_source = self.style_gan.mapping(style_source, None, truncation_psi=0.5, truncation_cutoff=8)
            latent_target = self.style_gan.mapping(style_target, None, truncation_psi=0.5, truncation_cutoff=8)
            sample_source =  self.style_gan.synthesis(latent_source, noise_mode='random', force_fp32=True)
            sample_target =  self.style_gan.synthesis(latent_target, noise_mode='random', force_fp32=True)
            
            sample_source = self.denorm(sample_source)
            sample_target = self.denorm(sample_target)
            emb_source = self.identity_embedder(F.interpolate(sample_source, size=256, mode='bilinear'))

            if self.latent_type == 'w':
                style_source = self.style_gan.style(style_source)
                style_target = self.style_gan.style(style_target)
        # endregion

        # region style, latent mixer
        if self.latent_type =='z' or self.latent_type == 'w':
            style_mix = style_target + self.identity_mixer(torch.cat((style_source, style_target), dim=1))
        
        else:
            latent_mix = self.identity_mixer(latent_source, latent_target)

        # endregion

        # region mixed latent to stylegan
        if self.latent_type =='z':
            sample_mix, _ = self.style_gan(
                [style_mix], truncation=1, truncation_latent=None, randomize_noise=True,
            )
        elif self.latent_type=='w':
            sample_mix, _ = self.style_gan(
            [style_mix], truncation=1, truncation_latent=None, randomize_noise=True, input_is_latent=True,
            )
        else:
            sample_mix = self.style_gan.synthesis(latent_mix, noise_mode='random', force_fp32=True)
            # sample_mix, _ = self.style_gan(
            #     [latent_mix], truncation=1, truncation_latent=None, randomize_noise=True, input_is_latent=True,
            # )
            style_target = latent_target
            style_mix = latent_mix
        sample_mix = self.denorm(sample_mix)        
        emb_mix = self.identity_embedder(F.interpolate(sample_mix, size=256, mode='bilinear'))
        # endregion
        
        g_loss, g_loss_dict = self.g_loss(sample_source, sample_target, sample_mix, emb_source, emb_mix, style_target, style_mix)
        image_dict = {}
        image_dict['I_target'] = sample_target[0:4]
        image_dict['I_source'] = sample_source[0:4]
        image_dict['I_swap'] = sample_mix[0:4]
        image_dict['I_diff'] = torch.abs(sample_target - sample_mix).clamp(0, 1)[0:4]

        return g_loss_dict, image_dict

    def training_step(self, batch, batch_idx):
        loss_dict, image_dict = self.common_step()

        # region logging
        self.logging_dict(loss_dict, prefix='train / ')
        self.logging_lr()
        if self.global_step % 1000 == 0:
            for key, val in image_dict.items():
                image_dict[key] = val.cpu()
            self.logging_image_dict(image_dict, prefix='train / ')
        # endregion

        return loss_dict['g_loss']


    def validation_step(self, batch, batch_idx):
        loss_dict, image_dict = self.common_step(batch_idx)
        self.logging_dict(loss_dict, prefix='validation / ')

        return image_dict

    def validation_epoch_end(self, outputs):
        val_images = []
        for idx, output in enumerate(outputs):
            if idx > 29:
                break
            val_images.append(output['I_source'][0].cpu())
            val_images.append(output['I_target'][0].cpu())
            val_images.append(output['I_swap'][0].cpu())
            val_images.append(output['I_diff'][0].cpu())
        
        val_image = torchvision.utils.make_grid(val_images, nrow=4)
        self.logger.experiment.log({'validation / img': wandb.Image(val_image.clamp(0, 1))}, commit=False)


    def configure_optimizers(self):
        optimizer_g = instantiate_from_config(self.hp.optimizer, params={"params": self.identity_mixer.parameters()})
        if "scheduler" in self.hp:
            scheduler_g = instantiate_from_config(self.hp.scheduler, params={"optimizer": optimizer_g})
            return {
                "optimizer": optimizer_g,
                "lr_scheduler": {
                    "scheduler": scheduler_g,
                    "interval": 'step',
                    "monitor": False,
                },
            }
        else:
            return {"optimizer": optimizer_g}

    def train_dataloader(self):
        trainset = EmptyDataset(10000)
        return DataLoader(trainset, batch_size=1)

    def val_dataloader(self):
        valset = EmptyDataset(100)
        return DataLoader(valset, batch_size=1)
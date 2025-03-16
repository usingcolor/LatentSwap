import torch
import torch.nn as nn
import torch.nn.functional as F
from .Deep3DFaceRecon_pytorch.models.bfm import ParametricFaceModel
from .Deep3DFaceRecon_pytorch.models.networks import ReconNetWrapper

class LatentSwapLoss(nn.Module):
    def __init__(self,id_coeff=1, ch_coeff=1, lp_coeff=10, shape_coeff=0.1, f_3d_checkpoint_path=None):
        super().__init__()
        self.id_coeff = id_coeff
        self.ch_coeff = ch_coeff
        self.lp_coeff = lp_coeff
        self.shape_coeff = shape_coeff

        self.f_3d = ReconNetWrapper(net_recon='resnet50', use_last_fc=False)
        self.f_3d.load_state_dict(torch.load(f_3d_checkpoint_path, map_location='cpu')['net_recon'])
        self.f_3d.eval()
        self.face_model = ParametricFaceModel()


    def identity_loss(self, id_vector1, id_vector2):
        inner_product_r = (torch.bmm(id_vector1.unsqueeze(1), id_vector2.unsqueeze(2)).squeeze())
        loss = (1 - inner_product_r).mean()
        return loss
    
    def latent_penalty(self, target_style, swap_style):
        return F.mse_loss(target_style, swap_style)
    
    def shape_loss(self, source_image, target_image, swap_image):
        with torch.no_grad():
            source_3dmm = self.f_3d(F.interpolate(source_image, size=224, mode='bilinear'))
            target_3dmm = self.f_3d(F.interpolate(target_image, size=224, mode='bilinear'))
            fuse_3dmm = torch.cat((source_3dmm[:, :80], target_3dmm[:, 80:]), dim=1)
            _, _, _, q_fuse = self.face_model.compute_for_render(fuse_3dmm)

        swap_3dmm = self.f_3d(F.interpolate(swap_image, size=224, mode='bilinear'))
        _, _, _, q_r = self.face_model.compute_for_render(swap_3dmm)
        return F.l1_loss(q_fuse, q_r)

    def forward(self, source_image, target_image, swap_image, source_embedding, swap_embedding, target_style, swap_style):
        id_loss = self.identity_loss(source_embedding, swap_embedding)
        lp_loss = self.latent_penalty(target_style, swap_style)
        shape_loss = self.shape_loss(source_image, target_image, swap_image)

        g_loss =self.id_coeff * id_loss + self.lp_coeff * lp_loss + self.shape_coeff * shape_loss
        return g_loss, {"g_loss": g_loss, "id_loss": id_loss, "lp_loss": lp_loss, "shape_loss": shape_loss}


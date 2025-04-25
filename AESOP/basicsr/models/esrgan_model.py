import torch
from collections import OrderedDict

from .srgan_model import SRGANModel

import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from thop import profile
import time
from ptflops import get_model_complexity_info

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import pyiqa

import torchvision.transforms.functional as F
import cv2
import numpy as np


@MODEL_REGISTRY.register()
class ESRGANModel(SRGANModel):
    """ESRGAN model for single image super-resolution."""

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss (relativistic gan)
            real_d_pred = self.net_d(self.gt).detach()
            fake_g_pred = self.net_d(self.output)
            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            l_g_gan = (l_g_real + l_g_fake) / 2

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # gan loss (relativistic gan)

        # In order to avoid the error in distributed training:
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # we separate the backwards for real and fake, and also detach the
        # tensor for calculating mean.

        # real
        fake_d_pred = self.net_d(self.output).detach()
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


@MODEL_REGISTRY.register()
class ESRGANModel_RealWorldTester(ESRGANModel):
    """
    Simple variant of ESRGAN model to manually add some degradation on flight.
    """
    
    def optimize_parameters(self, current_iter):
        raise NotImplementedError("This model is only for testing purposes.")
    
    def feed_data(self, data):
        tmp_lq = data['lq'].to(self.device)
        
        # ============== Degradation Steps ==============
        
        # Start with the original low-quality image tensor
        degraded = tmp_lq
        
        # 1) Add Gaussian noise
        degraded = degraded + torch.randn_like(tmp_lq) * 0.15
        
        # 2) Apply Gaussian blur
        degraded = F.gaussian_blur(degraded, kernel_size=(5, 5), sigma=(0.5, 0.5))
        
        # 3) Simulate JPEG compression (quality=50) with OpenCV in-memory
        degraded_np = degraded.squeeze(0).permute(1, 2, 0).cpu().numpy()
        degraded_np = np.clip(degraded_np * 255.0, 0, 255).astype(np.uint8)
        degraded_bgr = degraded_np[:, :, ::-1]
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, enc_img = cv2.imencode(".jpg", degraded_bgr, encode_params)
        decoded_bgr = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        decoded_rgb = decoded_bgr[:, :, ::-1].astype(np.float32) / 255.0
        degraded = torch.from_numpy(decoded_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Store the final degraded tensor
        self.lq = degraded
        
        # If ground truth is available, move it to the same device
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            

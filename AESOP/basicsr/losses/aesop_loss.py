import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
from basicsr.archs import build_network
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
from .losses import l1_loss, mse_loss

_reduction_modes = ['none', 'mean', 'sum']


@LOSS_REGISTRY.register()
class AutoEncoderLoss(nn.Module):
    """AutoEncoder loss.

    Args:

    """
    
    def __init__(self, criterion='l1', loss_weight=1.0, reduction='mean', autoencoder_arch=None, autoencoder_load=None, as_loss_map=False):
        
        super(AutoEncoderLoss, self).__init__()
        
        self.as_loss_map = as_loss_map
        self.loss_weight = loss_weight
        self.reduce = reduction
        
        # build and load autoencoder
        self.ae_net = build_network(autoencoder_arch)
        self.ae_load_path = autoencoder_load['path']
        self.ae_load_key = autoencoder_load['key']
        self.ae_net.load_state_dict(torch.load(self.ae_load_path)[self.ae_load_key])
        
        # freeze ae_net
        for param in self.ae_net.parameters():
            param.requires_grad = False
        
        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = l1_loss
        elif self.criterion_type == 'l2' or self.criterion_type == "mse":
            self.criterion = mse_loss
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')
    
    def forward(self, x, gt, weight=None):
        with torch.no_grad():
            gt_ae = self.ae_net(gt.detach())
        x_ae = self.ae_net(x)
        
        if self.as_loss_map:
            ae_weight = torch.abs(x_ae - gt_ae)
            return self.loss_weight * self.criterion(x, gt, weight * ae_weight, self.reduce)
        
        else:
            return self.loss_weight * self.criterion(x_ae, gt_ae, weight, self.reduce)


@LOSS_REGISTRY.register()
class AutoEncoderBottleneckLoss(nn.Module):
    """AutoEncoder loss.

    Args:

    """
    
    def __init__(self, criterion='l1', loss_weight=1.0, reduction='mean', autoencoder_arch=None, autoencoder_load=None, as_loss_map=False):
        
        super(AutoEncoderBottleneckLoss, self).__init__()
        
        self.as_loss_map = as_loss_map
        self.loss_weight = loss_weight
        self.reduce = reduction
        
        # build and load autoencoder
        self.ae_net = build_network(autoencoder_arch)
        self.ae_load_path = autoencoder_load['path']
        self.ae_load_key = autoencoder_load['key']
        self.ae_net.load_state_dict(torch.load(self.ae_load_path)[self.ae_load_key])
        
        # freeze ae_net
        for param in self.ae_net.parameters():
            param.requires_grad = False
        
        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = l1_loss
        elif self.criterion_type == 'l2' or self.criterion_type == "mse":
            self.criterion = mse_loss
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')
    
    def forward(self, x, gt, weight=None):
        with torch.no_grad():
            gt_ae, gt_bottleneck = self.ae_net(gt.detach(), return_bottleneck=True)
        x_ae, x_bottleneck = self.ae_net(x, return_bottleneck=True)
        
        if self.as_loss_map:
            raise NotImplementedError
        
        else:
            return self.loss_weight * self.criterion(x_bottleneck, gt_bottleneck, weight, self.reduce)


@LOSS_REGISTRY.register()
class ProbabilisticAutoEncoderReverseKLDivergenceLoss(nn.Module):
    
    def __init__(self, criterion='laplacian', loss_weight=1.0, reduction='mean', autoencoder_arch=None, autoencoder_load=None, as_loss_map=False):
        super(ProbabilisticAutoEncoderReverseKLDivergenceLoss, self).__init__()
        
        self.loss_weight = loss_weight
        
        # build and load autoencoder
        self.ae_net = build_network(autoencoder_arch)
        self.ae_load_path = autoencoder_load['path']
        self.ae_load_key = autoencoder_load['key']
        self.ae_net.load_state_dict(torch.load(self.ae_load_path)[self.ae_load_key])
        
        # freeze ae_net
        for param in self.ae_net.parameters():
            param.requires_grad = False
        
        self.criterion_type = criterion
        assert self.criterion_type in ['reverse', 'forward'], f'{criterion} criterion has not been supported.'
    
    def kl_divergence_laplacian(self, P_mu, P_b, Q_mu, Q_b):
        """
        Compute KL divergence between two Laplacian distributions.

        Args:
        - P_mu, P_b: Parameters for the first Laplacian distribution (P).
        - Q_mu, Q_b: Parameters for the second Laplacian distribution (Q).

        Returns:
        - KL divergence between the two distributions.
        """
        
        eps = 1e-6
        temperature = 2.0
        P_b = (torch.clamp(P_b, min=eps)) * temperature
        Q_b = (torch.clamp(Q_b, min=eps)) * temperature
        
        kl = torch.log(Q_b / P_b) + (P_b + torch.abs(P_mu - Q_mu)) / Q_b - 1
        return kl.mean()
    
    def forward(self, x, gt, weight=None):
        mu_sr, b_sr = self.ae_net(x, return_bottleneck=False, return_sigma=True)  # b is sigma for laplacian distribution
        mu_gt, b_gt = self.ae_net(gt.detach(), return_bottleneck=False, return_sigma=True)
        
        if self.criterion_type == "reverse":
            P_mu, P_b = mu_sr, b_sr
            Q_mu, Q_b = mu_gt, b_gt
        elif self.criterion_type == "forward":
            P_mu, P_b = mu_gt, b_gt
            Q_mu, Q_b = mu_sr, b_sr
        else:
            raise NotImplementedError(f'{self.criterion_type} criterion has not been supported.')
        
        kl_loss = self.kl_divergence_laplacian(P_mu, P_b, Q_mu, Q_b)
        
        return self.loss_weight * kl_loss


@LOSS_REGISTRY.register()
class LRL1Loss(nn.Module):
    """
    Simple version of L1 loss at LR scale / or equivalently, low-pass filtering
    """
    
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(LRL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')
        
        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        
        # bicubic downsample x4
        LR_pred = F.interpolate(pred, scale_factor=0.25, mode='bicubic', align_corners=False)
        LR_target = F.interpolate(target, scale_factor=0.25, mode='bicubic', align_corners=False)
        
        return self.loss_weight * l1_loss(LR_pred, LR_target, weight, reduction=self.reduction)

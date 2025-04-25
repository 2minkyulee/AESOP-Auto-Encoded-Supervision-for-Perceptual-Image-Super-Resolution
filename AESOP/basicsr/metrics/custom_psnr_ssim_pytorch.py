import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import tqdm
from .psnr_pytorch import psnr
import cv2
import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY
from .metric_util import quantize

def _ssim_pth(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).
    It is called by func:`calculate_ssim_pt`.
    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
    Returns:
        float: SSIM result.
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])

def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).
    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img


@METRIC_REGISTRY.register()
def calculate_ssim_pytorch(img1: torch.Tensor, img2: torch.Tensor, crop_border: int, test_y_channel: bool = False):
    """Calculate SSIM (structural similarity) (PyTorch version).
    ``Paper: Image quality assessment: From error visibility to structural similarity``
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img1 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: SSIM result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    img1, img2 = [quantize(img) for img in (img1, img2)]


    if crop_border != 0:
        img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img1 = rgb2ycbcr_pt(img1, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_pth(img1 * 255., img2 * 255.)
    ssim_score = ssim.mean(dim=0, keepdim=True)
    return ssim_score.squeeze().item()



@METRIC_REGISTRY.register()
def calculate_psnr_pytorch(img1: torch.Tensor, img2: torch.Tensor, crop_border: int, test_y_channel: bool = True):
    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    img1, img2 = [quantize(img) for img in (img1, img2)]
    
    if crop_border != 0:
        img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
    if test_y_channel:
        img1 = rgb2ycbcr_pt(img1, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)
    
    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)
    
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    psnr = 10. * torch.log10(1. / (mse + 1e-8))
    psnr_score = psnr.mean(dim=0, keepdim=True)
    
    return psnr_score.squeeze().item()


@METRIC_REGISTRY.register()
def calculate_aepsnr_pytorch(img1: torch.Tensor, img2: torch.Tensor, metric_module: nn.Module, crop_border: int, test_y_channel: bool = True):
    """
    Implementation of AutoEncoded PSNR.
    PSNR is calculated with Autoencoder(HR) and Autoencoder(SR) outputs.
    
    metric_module: Autoencoder model (nn.Module)
    
    """
    
    assert isinstance(metric_module, nn.Module), f'metric_module must be nn.Module, but got {type(metric_module)}'
    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    img1, img2 = [quantize(img) for img in (img1, img2)]  # must quantize first for fair comparison
    
    with torch.no_grad():  # just to make sure
        img1 = metric_module(img1.cuda())
        img2 = metric_module(img2.cuda())
    
    
    if crop_border != 0:
        img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
    if test_y_channel:
        img1 = rgb2ycbcr_pt(img1, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)
    
    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)
    
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    psnr = 10. * torch.log10(1. / (mse + 1e-8))
    psnr_score = psnr.mean(dim=0, keepdim=True)
    
    return psnr_score.squeeze().item()


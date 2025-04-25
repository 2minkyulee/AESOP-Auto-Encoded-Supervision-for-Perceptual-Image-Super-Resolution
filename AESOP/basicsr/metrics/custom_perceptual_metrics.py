import pyiqa
import cv2
import numpy as np
import torch
from basicsr.metrics.metric_util import reorder_image, to_y_channel, quantize
from basicsr.utils.registry import METRIC_REGISTRY
import pyiqa


# ignore warnings
import warnings
warnings.filterwarnings("ignore")



def common_np2pyiqa(img1, img2, crop_border, input_order='HWC'):
    # todo: directly convert to tensor

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # convert to pytorch tensor
    img1 = img1.transpose(2, 0, 1)
    img2 = img2.transpose(2, 0, 1)
    img1 = torch.from_numpy(img1).unsqueeze(0).float() / 255.0
    img2 = torch.from_numpy(img2).unsqueeze(0).float() / 255.0
    
    # BGR 2 RGB
    img1 = img1[:, [2, 1, 0], :, :]
    img2 = img2[:, [2, 1, 0], :, :]
    
    return img1, img2


@METRIC_REGISTRY.register()
def calculate_lpips(img1, img2, crop_border, metric_module, input_order='HWC'):
    """Calculate lpips

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the LPIPS calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: lpips result.
    """
    with torch.no_grad():
        assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
        img1, img2 = common_np2pyiqa(img1, img2, crop_border, input_order=input_order)
        
        lpips_module = metric_module
        lpips_score = lpips_module(img1.cuda(), img2.cuda()).squeeze().item()
    return lpips_score

@METRIC_REGISTRY.register()
def calculate_lpips_pytorch(img1, img2, crop_border, metric_module):
    """Calculate lpips pytorch

    Args:
        img1 (tensor): Images with range [0, 1].
        img2 (tensor): Images with range [0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the LPIPS calculation.
    Returns:
        float: lpips result.
    """
    with torch.no_grad():
        assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
        if crop_border != 0:
            img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img1, img2 = [quantize(img) for img in (img1, img2)]
        
        lpips_module = metric_module
        lpips_score = lpips_module(img1, img2).squeeze().item()

    return lpips_score


@METRIC_REGISTRY.register()
def calculate_dists(img1, img2, crop_border, metric_module, input_order='HWC'):
    """Calculate dists

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the DISTS calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: dists result.
    """
    with torch.no_grad():
        assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
        img1, img2 = common_np2pyiqa(img1, img2, crop_border, input_order=input_order)
        
        dists_module = metric_module
        dists_score = dists_module(img1.cuda(), img2.cuda()).squeeze().item()
    
        torch.set_grad_enabled(True)


@METRIC_REGISTRY.register()
def calculate_dists_pytorch(img1, img2, crop_border, metric_module):
    """Calculate dists pytorch

    Args:
        img1 (tensor): Images with range [0, 1].
        img2 (tensor): Images with range [0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the DISTS calculation.
    Returns:
        float: lpips result.
    """
    with torch.no_grad():
    
        assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
        img1, img2 = [quantize(img) for img in (img1, img2)]
        
        if crop_border != 0:
            img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
        dists_module = metric_module
        dists_score = dists_module(img1, img2).squeeze().item()
    return dists_score



@METRIC_REGISTRY.register()
def calculate_niqe_pytorch(img1, img2, crop_border, metric_module):
    with torch.no_grad():
        assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
        img1, img2 = [quantize(img) for img in (img1, img2)]
        
        if crop_border != 0:
            img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
        
        niqe_module = metric_module
        niqe_score = niqe_module(img1).squeeze().item()
    return niqe_score

@METRIC_REGISTRY.register()
def calculate_musiq_pytorch(img1, img2, crop_border, metric_module):
    with torch.no_grad():
        assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
        img1, img2 = [quantize(img) for img in (img1, img2)]
        
        if crop_border != 0:
            img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
        
        musiq_module = metric_module
        musiq_score = musiq_module(img1).squeeze().item()
        torch.set_grad_enabled(True)
    return musiq_score

@METRIC_REGISTRY.register()
def calculate_maniqa_pytorch(img1, img2, crop_border, metric_module):
    with torch.no_grad():
        assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
        img1, img2 = [quantize(img) for img in (img1, img2)]
        
        if crop_border != 0:
            img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
        
        maniqa_module = metric_module
        maniqa_score = maniqa_module(img1).squeeze().item()
        torch.set_grad_enabled(True)
    return maniqa_score


@METRIC_REGISTRY.register()
def calculate_clipiqa_resnet_pytorch(img1, img2, crop_border, metric_module):
    with torch.no_grad():
        assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
        img1, img2 = [quantize(img) for img in (img1, img2)]
        
        if crop_border != 0:
            img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
        
        clipiqa_module = metric_module
        clipiqa_score = clipiqa_module(img1, img2).squeeze().item()
        torch.set_grad_enabled(True)
    return clipiqa_score

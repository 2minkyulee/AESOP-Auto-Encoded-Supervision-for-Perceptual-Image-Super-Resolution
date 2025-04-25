from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim
from .custom_psnr_ssim_pytorch import calculate_ssim_pytorch, calculate_psnr_pytorch
from .custom_perceptual_metrics import calculate_lpips, calculate_dists, calculate_lpips_pytorch, calculate_dists_pytorch, calculate_niqe_pytorch, \
calculate_musiq_pytorch, calculate_maniqa_pytorch, calculate_clipiqa_resnet_pytorch


__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_lpips', 'calculate_dists',
           'calculate_psnr_pytorch', 'calculate_ssim_pytorch', 'calculate_lpips_pytorch', 'calculate_dists_pytorch',
           'calculate_niqe_pytorch', 'calculate_musiq_pytorch', 'calculate_maniqa_pytorch', 'calculate_clipiqa_resnet_pytorch']

def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must constain:
            type (str): Model type.
    """
    
    
    if "metric_module" not in opt.keys():
        opt = deepcopy(opt)
        metric_type = opt.pop('type')
        metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
        return metric
    
    else:
        # opt is oredered dict
        mm_tmp = opt.pop('metric_module')
        
        opt = deepcopy(opt)
        metric_type = opt.pop('type')
        
        # add mm_tmp back to opt
        opt['metric_module'] = mm_tmp
        
        metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
        return metric

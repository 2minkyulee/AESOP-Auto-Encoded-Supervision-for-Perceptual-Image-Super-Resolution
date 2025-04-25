import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import tqdm

from copy import deepcopy
from basicsr.archs.rrdbnet_arch import RRDBNet, ResidualDenseBlock, RRDB, make_layer
from basicsr.archs.srresnet_arch import MSRResNet
from basicsr.archs.arch_util import pixel_unshuffle, default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils import get_root_logger

@ARCH_REGISTRY.register()
class AutoEncoder_RRDBNet(nn.Module):
    """
    AutoEncoder architecture for AESOP loss
    """
    def __init__(self, enc_opt, dec_opt):
        super().__init__()
        
        # decoder
        dec_opt = deepcopy(dec_opt)
        enc_opt = deepcopy(enc_opt)
        
        # self.decoder = ARCH_REGISTRY.get(dec_opt.pop("type"))(**dec_opt)
        dec_opt.pop("type")
        self.decoder = RRDBNet(**dec_opt)  # only implemented for RRDBNet
        
        # encoder
        self.conv_first = nn.Sequential(
            nn.Conv2d(dec_opt["num_in_ch"], dec_opt["num_feat"]//16, 3, 1, 1),
            nn.Conv2d(dec_opt["num_feat"]//16, dec_opt["num_feat"]//16, 3, 1, 1),
        )
        self.down = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.PixelUnshuffle(2),
        )
        self.body = make_layer(RRDB, num_basic_block=2, num_feat=dec_opt["num_feat"], num_grow_ch=dec_opt["num_grow_ch"])
        self.conv_last = nn.Sequential(
            nn.Conv2d(dec_opt["num_feat"], dec_opt["num_feat"], 3, 1, 1),
            nn.Conv2d(dec_opt["num_feat"], dec_opt["num_in_ch"], 3, 1, 1),
        )
        
        
        # misc
        self.dec_is_frozen = False
        self.enc_is_frozen = False
        default_init_weights([self.conv_first, self.conv_last], 0.1)  # dont re-initiate rrdb weights. already done.
        self.encoder = nn.Sequential(
            self.conv_first,
            self.down,
            self.body,
            self.conv_last,
        )
        
        
    def freeze_encoder(self, current_iter="ITER_NOT_GIVEN"):
        if not self.enc_is_frozen:
            self.enc_is_frozen = True
            logger = get_root_logger()
            logger.info(f'Freeze encoder at {current_iter} iterations.')
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def freeze_decoder(self, current_iter="ITER_NOT_GIVEN"):
        if not self.dec_is_frozen:
            self.dec_is_frozen = True
            logger = get_root_logger()
            logger.info(f'Freeze decoder at {current_iter} iterations.')
            for param in self.decoder.parameters():
                param.requires_grad = False
    
    
    def unfreeze_encoder(self, current_iter="ITER_NOT_GIVEN"):
        if self.enc_is_frozen:
            self.enc_is_frozen = False
            logger = get_root_logger()
            logger.info(f'Unfreeze encoder at {current_iter} iterations.')
            for param in self.encoder.parameters():
                param.requires_grad = True
    
    def unfreeze_decoder(self, current_iter="ITER_NOT_GIVEN"):
        if self.dec_is_frozen:
            self.dec_is_frozen = False
            logger = get_root_logger()
            logger.info(f'Unfreeze decoder at {current_iter} iterations.')
            for param in self.decoder.parameters():
                param.requires_grad = True
            

        
    def forward(self, x, return_bottleneck=False):
        
        bottleneck = self.encoder(x)
        x = self.decoder(bottleneck)
        if return_bottleneck:
            return x, bottleneck
        else:
            return x
        
        
        
class AntialiasBicubicDown_x4(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=0.25, mode='bicubic', align_corners=False, antialias=True)



@ARCH_REGISTRY.register()
class BicubicDown_RRDBNet(AutoEncoder_RRDBNet):
    """
    AutoEncoder_RRDB
    Equal input output size
    For ablation purpose.
    """
    
    def __init__(self, enc_opt, dec_opt):
        super().__init__(enc_opt, dec_opt)
        
        # decoder
        self.dec_opt = deepcopy(dec_opt)
        self.enc_opt = deepcopy(enc_opt)
        
        # self.decoder = ARCH_REGISTRY.get(dec_opt.pop("type"))(**dec_opt)
        dec_opt.pop("type")
        self.decoder = RRDBNet(**dec_opt)
        
        # encoder
        # None
        
        # misc
        self.dec_is_frozen = False
        self.enc_is_frozen = False
        default_init_weights([self.conv_first, self.conv_last], 0.1)  # dont re-initiate rrdb weights. already done.
        self.encoder = nn.Sequential(
            # downsampling nn module
            AntialiasBicubicDown_x4()
        )
    
    def freeze_encoder(self, current_iter="ITER_NOT_GIVEN"):
        pass
    
    def freeze_decoder(self, current_iter="ITER_NOT_GIVEN"):
        if not self.dec_is_frozen:
            self.dec_is_frozen = True
            logger = get_root_logger()
            logger.info(f'Freeze decoder at {current_iter} iterations.')
            for param in self.decoder.parameters():
                param.requires_grad = False
    
    def unfreeze_encoder(self, current_iter="ITER_NOT_GIVEN"):
        pass
    
    def unfreeze_decoder(self, current_iter="ITER_NOT_GIVEN"):
        if self.dec_is_frozen:
            self.dec_is_frozen = False
            logger = get_root_logger()
            logger.info(f'Unfreeze decoder at {current_iter} iterations.')
            for param in self.decoder.parameters():
                param.requires_grad = True
    
    def forward(self, x, return_bottleneck=False):
        
        bottleneck = self.encoder(x)  # simply downsampling
        x = self.decoder(bottleneck)
        if return_bottleneck:
            return x, bottleneck
        else:
            return x
    
    
@ARCH_REGISTRY.register()
class ProbabilisticAutoEncoder_RRDBNet(AutoEncoder_RRDBNet):
    """
    Equivalent to AutoEncoder_RRDBNet but outputs mu and sigma instead of a single image (=mu).
    """
    def __init__(self, enc_opt, dec_opt):
        super().__init__(enc_opt, dec_opt)
        
        # additional sigma branch
        from basicsr.archs.rrdbnet_arch import RRDB
        self.sigma_branch = nn.Sequential(
            # body
            RRDB(num_feat=dec_opt["num_feat"], num_grow_ch=dec_opt["num_grow_ch"]),
            # upsample (following RRDB)
            nn.Conv2d(dec_opt["num_feat"], 3, 3, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # output convs (following RRDB)
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(3, 3, 3, 1, 1),
        )

        # apply forward hook on RRDBNet and register it
        # we follow namings from basicsr.archs.rrdbnet_arch.RRDBNet (conv_first, body)
        self.body_feat = None
        self.body_hook_handle = self.decoder.body.register_forward_hook(self._get_body_feat())
        
        self.conv_first_feat = None
        self.conv_first_hook_handle = self.decoder.body[0].register_forward_hook(self._get_first_conv_feat())
        
    
    def _get_body_feat(self):
        def _hook(module, input, output):
            self.body_feat = output.clone()
        return _hook
    
    def _get_first_conv_feat(self):
        def _hook(module, input, output):
            self.conv_first_feat = output.clone()
        return _hook
    
    
    def _remove_forward_hook(self):  # probably not needed?
        self.body_hook_handle.remove()
        self.conv_first_hook_handle.remove()
        
        
    def forward(self, x, return_bottleneck=False, return_sigma=False):
        
        bottleneck = self.encoder(x)
        x = self.decoder(bottleneck)  # features are now registered in self.body_feat and self.conv_first_feat
        
        feat = self.body_feat + self.conv_first_feat  # following RRDB, global skip connection
        sigma = self.sigma_branch(feat)
        
        
        # return
        output = [x]
        if return_bottleneck:
            output.append(bottleneck)
        if return_sigma:
            output.append(sigma)
            
        if len(output) == 1:
            output = output[0]
            
        return output


@ARCH_REGISTRY.register()
class AutoEncoder_MSRResNet(nn.Module):
    """
    AutoEncoder_MSRResNet
    Just for ablation + rebuttal .... @TODO: remove this part later
    """
    
    def __init__(self, enc_opt, dec_opt):
        super().__init__()
        
        # decoder
        dec_opt = deepcopy(dec_opt)
        enc_opt = deepcopy(enc_opt)
        
        # self.decoder = ARCH_REGISTRY.get(dec_opt.pop("type"))(**dec_opt)
        dec_opt.pop("type")
        self.decoder = MSRResNet(**dec_opt)
        
        # encoder
        self.conv_first = nn.Sequential(
            nn.Conv2d(dec_opt["num_in_ch"], 64 // 16, 3, 1, 1),
            nn.Conv2d(64 // 16, 64 // 16, 3, 1, 1),
        )
        self.down = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.PixelUnshuffle(2),
        )
        self.body = make_layer(RRDB, num_basic_block=2, num_feat=64, num_grow_ch=32)
        self.conv_last = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, dec_opt["num_in_ch"], 3, 1, 1),
        )
        
        # misc
        self.dec_is_frozen = False
        self.enc_is_frozen = False
        default_init_weights([self.conv_first, self.conv_last], 0.1)  # dont re-initiate rrdb weights. already done.
        self.encoder = nn.Sequential(
            self.conv_first,
            self.down,
            self.body,
            self.conv_last,
        )
    
    def freeze_encoder(self, current_iter="ITER_NOT_GIVEN"):
        if not self.enc_is_frozen:
            self.enc_is_frozen = True
            logger = get_root_logger()
            logger.info(f'Freeze encoder at {current_iter} iterations.')
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def freeze_decoder(self, current_iter="ITER_NOT_GIVEN"):
        if not self.dec_is_frozen:
            self.dec_is_frozen = True
            logger = get_root_logger()
            logger.info(f'Freeze decoder at {current_iter} iterations.')
            for param in self.decoder.parameters():
                param.requires_grad = False
    
    def unfreeze_encoder(self, current_iter="ITER_NOT_GIVEN"):
        if self.enc_is_frozen:
            self.enc_is_frozen = False
            logger = get_root_logger()
            logger.info(f'Unfreeze encoder at {current_iter} iterations.')
            for param in self.encoder.parameters():
                param.requires_grad = True
    
    def unfreeze_decoder(self, current_iter="ITER_NOT_GIVEN"):
        if self.dec_is_frozen:
            self.dec_is_frozen = False
            logger = get_root_logger()
            logger.info(f'Unfreeze decoder at {current_iter} iterations.')
            for param in self.decoder.parameters():
                param.requires_grad = True
    
    def forward(self, x, return_bottleneck=False):
        
        bottleneck = self.encoder(x)
        x = self.decoder(bottleneck)
        if return_bottleneck:
            return x, bottleneck
        else:
            return x
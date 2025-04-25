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
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from .sr_model import SRModel


from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
import random
import numpy as np
import torch.nn.functional as F





@MODEL_REGISTRY.register()
class AEModel(SRModel):

    def init_training_settings(self):
        """
        Changes from SRModel:
        0. load the decoder (if specified. Note that the decoder should not be loaded if the generator is loaded or vice versa)
        1. dont allow perceptual loss
        2. freeze or unfreeze the decoder
        """
        
        load_path_gen = self.opt['path'].get('pretrain_network_g', None)  # generator here means the total auto-encoder
        load_key_gen = self.opt['path'].get('param_key_g', None)
        load_path_dec = self.opt['path'].get('pretrain_network_decoder', None)
        load_key_dec = self.opt['path'].get('param_key_decoder', None)
        assert not (load_path_gen is not None and load_path_dec is not None), 'Cannot load both the generator and decoder.'
        
        if load_path_dec is not None:
            self.load_network(self.get_bare_model(self.net_g).decoder, load_path_dec, self.opt['path'].get('strict_load_decoder', True), load_key_dec)
            logger = get_root_logger()
            logger.info(f'** Loaded decoder from {load_path_dec}')
        
        if load_path_gen is not None:
            self.load_network(self.net_g, load_path_gen, self.opt['path'].get('strict_load_g', True), load_key_gen)
            logger = get_root_logger()
            logger.info(f'** Loaded generator from {load_path_gen}')
        
        
        # ------------------
        
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            for p in self.net_g_ema.parameters():
                p.requires_grad = False
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('pixel_at_LR_opt'):
            self.cri_pix_at_LR = build_loss(train_opt['pixel_at_LR_opt']).to(self.device)
        else:
            self.cri_pix_at_LR = None




        if train_opt.get('perceptual_opt'):
            raise ValueError('Perceptual loss should be none.')
            # self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        # freeze dec
        self.freeze_dec_until = train_opt.get('freeze_decoder', 0)
        if self.freeze_dec_until > 0:
            self.get_bare_model(self.net_g).freeze_decoder()
            


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        if 'gt' not in data:
            raise ValueError('GT does not exist in current dataset. GT must exists since it is an Auto-Encoder.')

    def optimize_parameters(self, current_iter):

        # unfreeze decoder
        if self.freeze_dec_until < current_iter:
            self.get_bare_model(self.net_g).unfreeze_decoder()
            

        # start
        self.optimizer_g.zero_grad()
        self.output, self.bottleneck = self.net_g(self.gt, return_bottleneck=True)  # forward gt

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        
        if self.cri_pix_at_LR:
            l_pix_at_LR = self.cri_pix_at_LR(self.bottleneck, self.lq)
            l_total += l_pix_at_LR
            loss_dict['l_pix_at_LR'] = l_pix_at_LR
        
        # perceptual loss
        if self.cri_perceptual:
            raise ValueError('Perceptual loss should be None.')
            
            
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.gt)  # forward gt
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.gt) # forward gt
            self.net_g.train()


@MODEL_REGISTRY.register()
class ProbabilisticAEModel(AEModel):
    
    def init_training_settings(self):

        super().init_training_settings()
        train_opt = self.opt['train']
        
        # define losses
        if train_opt.get('sigmabranch_opt'):
            self.cri_sigmabranch = build_loss(train_opt['sigmabranch_opt']).to(self.device)
        else:
            self.cri_sigmabranch = None
        
        
        
        # freeze dec
        self.freeze_dec_until = train_opt.get('freeze_decoder', 0)
        if self.freeze_dec_until > 0:
            print('Freeze decoder until iter %d' % self.freeze_dec_until)
            self.get_bare_model(self.net_g).freeze_decoder()
    
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        if 'gt' not in data:
            raise ValueError('GT does not exist in current dataset. GT must exists since it is an Auto-Encoder.')
    
    def optimize_parameters(self, current_iter):
        
        # unfreeze decoder
        if self.freeze_dec_until < current_iter:
            self.get_bare_model(self.net_g).unfreeze_decoder()
        
        # start
        self.optimizer_g.zero_grad()
        self.output, self.bottleneck, self.sigma_output = self.net_g(self.gt, return_bottleneck=True, return_sigma=True)  # forward gt
        
        
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        
        if self.cri_pix_at_LR:
            l_pix_at_LR = self.cri_pix_at_LR(self.bottleneck, self.lq)
            l_total += l_pix_at_LR
            loss_dict['l_pix_at_LR'] = l_pix_at_LR
        
        if self.cri_sigmabranch:
            l_sigmabranch = self.cri_sigmabranch(self.sigma_output, torch.abs(self.output.detach()-self.gt))
            l_total += l_sigmabranch
            loss_dict['l_sigmabranch'] = l_sigmabranch
        
        # perceptual loss
        if self.cri_perceptual:
            raise ValueError('Perceptual loss should be None.')
        
        l_total.backward()
        self.optimizer_g.step()
        
        self.log_dict = self.reduce_loss_dict(loss_dict)
        
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    def test(self):  # tmp code in order to visualize sigma... will be never used in real test


        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net = self.net_g_ema
        else:
            self.net_g.eval()
            net = self.net_g

        with torch.no_grad():

            bottleneck = net.encoder(self.gt)
            x = net.decoder(bottleneck)  # features are now registered in self.body_feat and self.conv_first_feat
            feat = net.body_feat + net.conv_first_feat  # following RRDB, global skip connection
            sigma = net.sigma_branch(feat)

            self.output = sigma

            # normalize to 0-1
            self.output = (self.output - self.output.min()) / (self.output.max() - self.output.min())
            self.output = self.output.clamp(0, 1)

            print("NEVER USE THIS FOR GENERAL TEST. THIS IS JUST FOR VISUALIZING SIGMA.")
            print(self.output.shape)
    

@MODEL_REGISTRY.register()
class RealWorldAEModel(AEModel):
    def __init__(self, opt):
        super(RealWorldAEModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        self.queue_size = opt['queue_size']
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        # training pair pool
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, 'queue size should be divisible by batch size'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()
            
            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b
    
    @torch.no_grad()
    def feed_data(self, data):
        if self.is_train:
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            # USM the GT images
            if self.opt['gt_usm'] is True:
                self.gt = self.usm_sharpener(self.gt)
            
            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)
            
            ori_h, ori_w = self.gt.size()[2:4]
            
            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            
            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            
            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
            
            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            
            # random crop
            gt_size = self.opt['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])
            
            # training pair pool
            self._dequeue_and_enqueue()
        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
    
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealWorldAEModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True


# [CVPR2025] AESOP 🦊🍇
### ⭐This is the official repository of
> **Auto-Encoded Supervision for Perceptual Image Super-Resolution ([Arxiv](https://arxiv.org/abs/2412.00124) | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Auto-Encoded_Supervision_for_Perceptual_Image_Super-Resolution_CVPR_2025_paper.pdf) | [Supplementary Material](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Lee_Auto-Encoded_Supervision_for_CVPR_2025_supplemental.pdf))**\
> MinKyu Lee, Sangeek Hyun, Woojin Jun, Jae-Pil Heo*\
Sungkyunkwan University\
\*: Corresponding Author


### Abstract
> This work tackles the fidelity objective in the perceptual super-resolution (SR). Specifically, we address the shortcomings of pixel-level $L_\text{p}$ loss ($L_\text{pix}$) in the GAN-based SR framework. Since $L_\text{pix}$ is known to have a trade-off relationship against perceptual quality, prior methods often multiply a small scale factor or utilize low-pass filters. However, this work shows that these circumventions fail to address the fundamental factor that induces blurring. Accordingly, we focus on two points: 1) precisely discriminating the subcomponent of $L_\text{pix}$ that contributes to blurring, and 2) only guiding based on the factor that is free from this trade-off relationship. We show that they can be achieved in a surprisingly simple manner, with an Auto-Encoder (AE) pretrained with $L_\text{pix}$. Accordingly, we propose the Auto-Encoded Supervision for Optimal Penalization loss ($L_\text{AESOP}$), a novel loss function that measures distance in the AE space, instead of the raw pixel space. Note that the AE space indicates the space after the decoder, not the bottleneck. By simply substituting $L_\text{pix}$ with $L_\text{AESOP}$, we can provide effective reconstruction guidance without compromising perceptual quality. Designed for simplicity, our method enables easy integration into existing SR frameworks. Experimental results verify that AESOP can lead to favorable results in the perceptual SR task.


### 🦊 **Check out our recent works:**
[[Arxiv]](https://arxiv.org/abs/2504.06629), [[Github]](https://github.com/2minkyulee/Analyzing-the-Training-Dynamics-of-Image-Restoration-Transformers) Analyzing the Training Dynamics of Image Restoration Transformers: A Revisit to Layer Normalization
\
[[Arxiv]](https://arxiv.org/abs/2312.17526), [[Github]](https://github.com/2minkyulee/Noise-free-Optimization-in-Early-Training-Steps-for-Image-Super-Resolution) (AAAI2024) Noise-free Optimization in Early Training Steps for Image Super-Resolution



## News
- 2024-12-04: 🎉 **Repository created!**
- 2025-02-07: 🎉 **Our paper has been accepted to CVPR2025!**  
- 2025-04-25: 🎉 **Codes updated!**

------

## Getting Started
1. Clone this repository
2. Setup environment via below.
```bash
bash custom_setup.sh
```
3. Download train/test datasets (and preprocess if required). Refer to instructions from [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md).
4. Download pre-trained weights from [Google Drive](https://drive.google.com/drive/folders/1eTHcQXD8kI9nK6IjQ8DidA0qJbJZkeZ_?usp=sharing)

------

## Tips
Refer to below for AESOP-relevant codes.
Direct copy-paste into other standard basicsr-based projects will work.

1. AESOP/basicsr/archs/autoencoder_arch.py (class AutoEncoder_RRDBNet)
2. AESOP/basicsr/models/aesop_esrganArtifactsDis_model.py (class AesopESRGANArtifactsDisModel)
3. AESOP/basicsr/losses/aesop_loss.py (class AutoEncoderLoss)
4. AESOP/options/train/AESOP

------

## Testing the SR network
```bash
# Make sure to modify the options below in the config file.
# 1. the test dataset path (dataroot_gt, dataroot_lq) 
# 2. the pretrained SR network weight path to test (path.pretrain_network_g) 

PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/AESOP/main/test_Synthetic_AESOP_RRDB128.yml
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/AESOP/main/test_Synthetic_AESOP_RRDB256.yml
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/AESOP/main/test_Synthetic_AESOP_SwinIR256.yml
```
------


## Training the SR network

```bash
# Make sure to modify the options below in the config file.
# 1. the train/val dataset paths (dataroot_gt, dataroot_lq)
# 2. the pretrained AutoEncoder path (train.aesop_opt.autoencoder_load.path), used for the AESOP loss 
# 3. the pretrained "PSNR-oriented" SR network weight path (path.pretrain_network_g), used for initializing the SR network 

PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=5678 basicsr/train.py -opt options/train/AESOP/train_Synthetic_AESOP_RRDB.yml --launcher pytorch
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=5678 basicsr/train.py -opt options/train/AESOP/train_Synthetic_AESOP_SwinIR.yml --launcher pytorch

# To resume,
# 1. additionally set a "--auto_resume" flag
# 2. and also make sure to modify the wandb log id in the config file (wandb.logger.resume_id)
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=5678 basicsr/train.py -opt options/train/AESOP/train_Synthetic_AESOP_RRDB.yml --launcher pytorch --auto_resume
```

------


## Training the AutoEncoder

```bash
# Make sure to modify the options below in the config file.
# 1. the train/val dataset paths (dataroot_gt, dataroot_lq)
# 2. the pretrained PSNR-oriented SR network weight path (path.pretrain_network_decoder), used for initializing the decoder 

PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=5678 basicsr/train.py -opt options/train/AutoEncoder/train_Synthetic_AE_RRDB_LRrecon1.yml --launcher pytorch
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=5678 basicsr/train.py -opt options/train/AutoEncoder/train_Realworld_AE_RRDB_DecoderFreeze.yml --launcher pytorch

```


------

## Acknowledgement
This project is built based on [LDL](https://github.com/csjliang/LDL), [BasicSR](https://github.com/XPixelGroup/BasicSR) and also
[DRCT](https://github.com/ming053l/drct),
[SwinIR](https://github.com/cszn/KAIR/tree/master)

------

## License
This project is released under the Apache 2.0 license.

------


## Contact
Please contact me via 2minkyulee@gmail.com for any inquiries.

------
## Citation
```
@article{lee2024auto,
  title={Auto-Encoded Supervision for Perceptual Image Super-Resolution},
  author={Lee, MinKyu and Hyun, Sangeek and Jun, Woojin and Heo, Jae-Pil},
  journal={arXiv preprint arXiv:2412.00124},
  year={2024}
}
```

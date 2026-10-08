## This function is adapted from RoSteALS and LaWa paper
import numpy as np
import einops
import torch
import torch.nn as nn
# from torch.nn import functional as thf
import pytorch_lightning as pl
import torchvision
from ldm.modules.diffusionmodules.util import (
    zero_module,
)
import os
import copy
# from contextlib import contextmanager
# from torchvision.utils import make_grid
# from ldm.modules.attention import SpatialTransformer
# from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
# from ldm.models.diffusion.ddpm import LatentDiffusion
# from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.util import instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.modules.ema import LitEma
# from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
# from ldm.modules.diffusionmodules.model import Encoder
import lpips
import torch.nn.functional as F
# from torch.autograd.variable import Variable

from torchvision import transforms
# from torchvision import models
import random
from lpips.lpips import LPIPS

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class NoiseSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, noise_module, global_step, active_step, p):

        ctx.original_shape = img.shape

        with torch.no_grad():
            noised_img = noise_module(img.clone(), global_step, active_step=active_step, p=p)
        return noised_img

    @staticmethod
    def backward(ctx, grad_output):
        orig_shape = ctx.original_shape


        if grad_output.shape != orig_shape:

            grad_input = F.interpolate(
                grad_output,
                size=(orig_shape[2], orig_shape[3]),
                mode='bilinear',
                align_corners=False
            )
        else:
            grad_input = grad_output

        return grad_input, None, None, None, None


class R2L(pl.LightningModule):
    def __init__(self,
                 first_stage_config,
                 decoder_config,
                 discriminator_config,
                 recon_type,
                 learning_rate=0.0001,
                 epoch_num=100,
                 recon_loss_weight=3.0,
                 adversarial_loss_weight=3.0,
                 perceptual_loss_weight=3.0,
                 lpips_loss_weights_path=None,
                 message_absolute_loss_weight=None,
                 ramp=100000,
                 watermark_addition_weight=0.1,
                 noise_config='__none__',
                 use_ema=False,
                 scale_factor=1.,
                 ckpt_path="__none__",
                 extraction_resize=False,
                 addition_network_config=None,
                 start_attack_acc_thresh=0.96,
                 dis_update_freq=1,
                 clamp_during_training=False,
                 noise_block_size=8,
                 latent_noise_dir=None,
                 ):
        super().__init__()
        self.latent_noise_dir = latent_noise_dir
        if self.latent_noise_dir:
            print(f"Running with Latent Noise from: {self.latent_noise_dir}")
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.scale_factor = scale_factor
        self.extraction_resize = extraction_resize
        self.ae = instantiate_from_config(first_stage_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.discriminator = instantiate_from_config(discriminator_config)

        if addition_network_config != None:
            self.addition_net = instantiate_from_config(addition_network_config)
        else:
            self.addition_net = None

        self.decoder_latent = instantiate_from_config(decoder_latent_config)

        if noise_config != '__none__':
            print('Using noise')
            self.noise = instantiate_from_config(noise_config)

        # copy weights from first stage
        # freeze first stage
        self.ae.eval()
        self.ae.train = disabled_train
        for p in self.ae.parameters():
            p.requires_grad = False

        self.watermark_addition_weight = watermark_addition_weight

        # early training phase
        self.message_len = decoder_config.params.message_len
        self.fixed_x = None
        self.fixed_img = None
        self.fixed_input_recon = None
        self.fixed_control = None
        self.fixed_latent_noise = None
        self.register_buffer("fixed_input", torch.tensor(False))
        self.register_buffer("noise_activated", torch.tensor(False))
        self.noise_active_step = 0
        self.ramp_step = 1e9
        self.start_attack_acc_thresh = start_attack_acc_thresh
        self.dis_update_freq = dis_update_freq
        self.tanh_activation = nn.Tanh()
        self.use_ema = use_ema

        if ckpt_path != '__none__':
            print("###############################################################################")
            print("Using provided model weights!")
            self.init_from_ckpt(ckpt_path, ignore_keys=[])

        #### image normalization parameters:
        self.normalize_vqgan_to_imagenet = transforms.Compose(
            [transforms.Normalize(mean=[-1, -1, -1], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        #### loss parameters:
        self.recon_type = recon_type
        if self.recon_type == 'yuv':
            self.register_buffer('yuv_scales', torch.tensor([1, 100, 100]).unsqueeze(1).float())  # [3,1]

        self.recon_weight = recon_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight

        if lpips_loss_weights_path != None:
            self.perceptual_loss = LPIPS(weights_path=lpips_loss_weights_path)
            self.perceptual_loss.eval()
        elif self.perceptual_loss_weight > 0 and lpips_loss_weights_path == None:
            self.perceptual_loss = LPIPS()
            self.perceptual_loss.eval()

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.adversarial_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.message_absolute_loss_weight = message_absolute_loss_weight

        ### watermark layers:
        # watermark
        self.noise_block_size = noise_block_size

        # --- 4 -> 3 ---
        self.watermark_initial_0 = nn.Sequential(
            nn.Linear(self.message_len, 3 * self.noise_block_size * self.noise_block_size),
            nn.SiLU(),
            View(-1, 3, self.noise_block_size, self.noise_block_size),
        )
        for p in self.watermark_initial_0.parameters():
            p.requires_grad = True

        self.watermark_initial_0_conv = zero_module(torch.nn.Conv2d(3, 3, 3, padding=1))
        for p in self.watermark_initial_0_conv.parameters():
            p.requires_grad = True

        # --- Middle Block---
        self.watermark_initial = nn.Sequential(
            nn.Linear(self.message_len, 512 * self.noise_block_size * self.noise_block_size),
            nn.SiLU(),
            View(-1, 512, self.noise_block_size, self.noise_block_size),
        )
        for p in self.watermark_initial.parameters():
            p.requires_grad = True

        self.watermark_initial_conv = zero_module(torch.nn.Conv2d(512, 512, 3, padding=1))
        for p in self.watermark_initial_conv.parameters():
            p.requires_grad = True

        self.watermark = nn.ModuleList()
        self.watermark_conv = nn.ModuleList()

        # --- Layer 4---
        layer_4 = nn.Sequential(
            nn.Linear(self.message_len, 128 * self.noise_block_size * self.noise_block_size),
            nn.SiLU(),
            View(-1, 128, self.noise_block_size, self.noise_block_size),
        )
        for p in layer_4.parameters():
            p.requires_grad = True
        self.watermark.append(layer_4)

        layer_4_conv = zero_module(torch.nn.Conv2d(128, 128, 3, padding=1))
        for p in layer_4_conv.parameters():
            p.requires_grad = True
        self.watermark_conv.append(layer_4_conv)

        # --- Layer 3---
        layer_3 = nn.Sequential(
            nn.Linear(self.message_len, 256 * self.noise_block_size * self.noise_block_size),
            nn.SiLU(),
            View(-1, 256, self.noise_block_size, self.noise_block_size),
        )
        for p in layer_3.parameters():
            p.requires_grad = True
        self.watermark.append(layer_3)

        layer_3_conv = zero_module(torch.nn.Conv2d(256, 256, 3, padding=1))
        for p in layer_3_conv.parameters():
            p.requires_grad = True
        self.watermark_conv.append(layer_3_conv)

        # --- Layer 2  ---
        layer_2 = nn.Sequential(
            nn.Linear(self.message_len, 512 * self.noise_block_size * self.noise_block_size),
            nn.SiLU(),
            View(-1, 512, self.noise_block_size, self.noise_block_size),
        )
        for p in layer_2.parameters():
            p.requires_grad = True
        self.watermark.append(layer_2)

        layer_2_conv = zero_module(torch.nn.Conv2d(512, 512, 3, padding=1))
        for p in layer_2_conv.parameters():
            p.requires_grad = True
        self.watermark_conv.append(layer_2_conv)



    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

   
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.control_ema(self.control)
            self.decoder_ema(self.decoder)

    # You can edit the code here based on the outline we proposed.
    # def forward(self, x, image, c):
    # def get_input(self, batch, bs=None):
    # def training_step(self, batch, batch_idx, optimizer_idx):



    def on_train_epoch_end(self):

        metrics = self.trainer.callback_metrics


        avg_acc = metrics.get("train/bit_acc", torch.tensor(0.0)).item()
        avg_psnr = metrics.get("train/psnr", torch.tensor(0.0)).item()

        epoch_num = self.current_epoch


        print(f"\n========================================================")
        print(f"  Epoch {epoch_num} Finished!")
        print(f"  ---------------------")
        print(f"  Avg Acc:  {avg_acc:.4f}  ({avg_acc * 100:.2f}%)")
        print(f"  Avg PSNR: {avg_psnr:.2f} dB")
        print(f"========================================================\n")


    # def validation_step(self, batch, batch_idx):



    def configure_optimizers(self):

        embedding_params = list(self.watermark_initial_0.parameters()) + \
                           list(self.watermark_initial_0_conv.parameters()) + \
                           list(self.watermark_initial.parameters()) + \
                           list(self.watermark_initial_conv.parameters()) + \
                           list(self.watermark[0].parameters()) + \
                           list(self.watermark_conv[0].parameters()) + \
                           list(self.watermark[1].parameters()) + \
                           list(self.watermark_conv[1].parameters()) + \
                           list(self.watermark[2].parameters()) + \
                           list(self.watermark_conv[2].parameters()) + list(self.decoder.parameters())

        discriminator_params = list(self.discriminator.parameters())

        embedding_optimizer = torch.optim.AdamW(embedding_params, lr=self.learning_rate)
        discriminator_optimizer = torch.optim.AdamW(discriminator_params, lr=self.learning_rate)

        embedding_lr_scheduler = CosineAnnealingLR(embedding_optimizer, T_max=self.epoch_num)
        discriminator_lr_scheduler = CosineAnnealingLR(discriminator_optimizer, T_max=self.epoch_num)

        if self.dis_update_freq == 0:
            return [embedding_optimizer, discriminator_optimizer]
        elif self.dis_update_freq > 0:
            return [
                {
                    "optimizer": embedding_optimizer,
                    "frequency": 1,
                    # "lr_scheduler": embedding_lr_scheduler,
                },
                {
                    "optimizer": discriminator_optimizer,
                    "frequency": self.dis_update_freq,
                    # "lr_scheduler": discriminator_lr_scheduler,
                },
            ]

    def log_images(self, batch, fixed_input=False, **kwargs):
        with torch.no_grad():
            log = dict()
            if fixed_input and self.fixed_img is not None:
                x, c, img, img_recon, latent_noise = self.fixed_x, self.fixed_control, self.fixed_img, self.fixed_input_recon, self.fixed_latent_noise
            else:
                x, c, img, img_recon, latent_noise = self.get_input(batch)

            x, image_out = self(x, img_recon, c)
            y = self.encode_first_stage(image_out)
            y = y + latent_noise
            y = self.ae.post_quant_conv(1. / self.scale_factor * y)
            watermark_noise_image = self.ae.decoder(y)

            watermark_noise_image = torch.clamp(watermark_noise_image, min=-1., max=1.)

            if hasattr(self, 'noise') and self.noise_activated:
                img_noise = self.noise(watermark_noise_image, self.global_step, p=1.0)
                log['noised'] = img_noise
            log['input'] = img
            log['output'] = image_out
            log['recon'] = img_recon
            return log



    
    ### image to feature and feature to image functions:
    def decode_first_stage(self, z):
        z = 1./self.scale_factor * z
        image_rec = self.ae.decode(z)
        return image_rec

    def decode_first_stage_watermark(self, z):
        z = 1./self.scale_factor * z
        delta_I = self.ae_watermark_decoder(z)
        delta_I = self.tanh_activation(delta_I)
        return delta_I
    

    def encode_first_stage(self, x):
        encoder_posterior = self.ae.encode(x)

        if isinstance(encoder_posterior, tuple):
            z = encoder_posterior[0]

        elif hasattr(encoder_posterior, "sample"):
            z = encoder_posterior.sample()

        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")

        return self.scale_factor * z



    ### loss functions:
    def compute_recon_loss(self, inputs, reconstructions):
        if self.recon_type == 'rgb':
            # rec_loss = torch.abs(inputs - reconstructions).mean(dim=[1,2,3])
            # rec_loss = torch.mean((inputs - reconstructions)**2, dim=[1,2,3])
            rec_loss = torch.mean((inputs - reconstructions)**2)
        elif self.recon_type == 'yuv':
             
            reconstructions_yuv = self.rgb_to_yuv((reconstructions + 1) / 2)
            inputs_yuv = self.rgb_to_yuv((inputs + 1) / 2)
            yuv_loss = torch.mean((reconstructions_yuv - inputs_yuv)**2, dim=[2,3])
            rec_loss = torch.mean(torch.mm(yuv_loss, self.yuv_scales))
            
        elif self.recon_type == 'watson_vgg':
            rec_loss = self.loss_w_vgg((1+inputs)/2.0, (1+reconstructions)/2.0) / reconstructions.shape[0]   
        else:
            raise ValueError(f"Unknown recon type {self.recon_type}")
        return rec_loss
    
    
    def calculate_psnr(self, image_rec, img_rec_gt):
        ## calculate psnr:
        with torch.no_grad():
            delta = 255 * torch.clamp((image_rec+1.0) / 2.0 - (img_rec_gt+1.0) / 2.0, 0 , 1)
            delta = delta.reshape(-1, image_rec.shape[-3], image_rec.shape[-2], image_rec.shape[-1]) # BxCxHxW
            psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2, dim=(1,2,3)))
            psnr = psnr.mean()
        return psnr
    
    def rgb_to_yuv(self, image):    
        r = image[..., 0, :, :]
        g = image[..., 1, :, :]
        b = image[..., 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v = 0.615 * r - 0.515 * g - 0.100 * b

        return torch.stack([y, u, v], -3)
    
    def nonlinearity(self,x):
        # swish
        return x*torch.sigmoid(x)


class Discriminator1(nn.Module):
    """
    Discriminator network to differentiate between watermarked and non-watermarked images
    """
    def __init__(self):
        super(Discriminator1, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        self.head = nn.Linear(128, 1)

    def copy_encoder_weight(self, ae_model):
        return None
    
    def forward(self, image):
        x = self.layers(image) # (B,C,H,W) --> (B,C,1,1)
        x.squeeze_(3).squeeze_(2)
        x = self.head(x)
        return x



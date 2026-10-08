# import os
# import torch
# import numpy as np
# from torch import nn
# import random
# import torch.nn.functional as F
# from PIL import Image
# from torchvision.transforms import functional
# from augly.image import functional as augly_functional
# from torchvision import transforms
# from kornia.geometry.transform import get_rotation_matrix2d, warp_affine
# import kornia
#
# def _compute_translation_matrix_single(dx: float, dy: float, device='cuda'):
#     M = torch.tensor([[[1, 0, dx],
#                        [0, 1, dy]]], device=device, dtype=torch.float32)
#     return M
# def get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size):
#     rnd_hue = torch.FloatTensor(batch_size, 3, 1, 1).uniform_(-rnd_hue, rnd_hue)
#     rnd_brightness = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(-rnd_bri, rnd_bri)
#     return rnd_hue, rnd_brightness
#
#
# class TransformNet(nn.Module):
#     """"
#     This class is adapted from RoSteALS paper
#     """
#     def __init__(self, rnd_bri=0.3, rnd_hue=0.1, rnd_noise=0.02, rnd_sat=1.0, rnd_trans=0.1,contrast=[0.5, 1.5], ramp=1000,
#                  apply_many_crops = False, apply_required_attacks = False, required_attack_list = ['resize','random_crop']) -> None:
#         super().__init__()
#         self.rnd_bri = rnd_bri
#         self.rnd_hue = rnd_hue
#         self.rnd_noise = rnd_noise
#         self.rnd_sat = rnd_sat
#         self.rnd_trans = rnd_trans
#         self.contrast_low, self.contrast_high = contrast
#         self.ramp = ramp
#         self.apply_many_crops = apply_many_crops
#         self.apply_required_attacks = apply_required_attacks
#         self.register_buffer('step0', torch.tensor(0))  # large number
#         self.required_attack_list = required_attack_list
#         self.imagenet_mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1), requires_grad=False)
#         self.imagenet_std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1), requires_grad = False)
#
#     def imgnet_unnormalize(self,x):
#         return (x * self.imagenet_std) + self.imagenet_mean
#
#     def imgnet_normalize(self,x):
#         return (x - self.imagenet_mean) / self.imagenet_std
#
#     def jpeg_compress(self, x, quality_factor):
#         """ jpeg code from Stable Signature"""
#         """ Apply jpeg compression to image
#         Args:
#             x: normalized Tensor image between [-1, 1]
#             quality_factor: quality factor
#         """
#         with torch.no_grad():
#             to_pil = transforms.ToPILImage()
#             to_tensor = transforms.ToTensor()
#             img_jpeg = torch.zeros_like(x, device=x.device)
#             ## clampt the values:
#             x_clip = torch.round(255 * self.imgnet_unnormalize(x)).clamp(0, 255) / 255.0
#             for ii,img in enumerate(x_clip):
#                 pil_img = to_pil(img)
#                 img_jpeg[ii] = to_tensor(augly_functional.encoding_quality(pil_img, quality=quality_factor))
#             img_gap = self.imgnet_normalize(img_jpeg) - x
#             img_gap = img_gap.detach()
#         img_jpeg_compressed_differentiable = x + img_gap
#         return img_jpeg_compressed_differentiable
#
#
#     def forward(self, x, global_step, active_step = 0, p=0.9):
#
#         if torch.rand(1)[0] >= p:
#             return x
#
#         batch_size, sh, device = x.shape[0], x.size(), x.device
#         if active_step == 0:
#             ramp_fn = lambda ramp: np.min([(global_step-self.step0.cpu().item()) / ramp, 1.])
#         else:
#             ramp_fn = lambda ramp: np.min([(global_step-active_step) / ramp, 1.])
#
#         rnd_bri = ramp_fn(self.ramp) * self.rnd_bri
#         rnd_hue = ramp_fn(self.ramp) * self.rnd_hue
#         rnd_brightness = get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size).to(device)  # [batch_size, 3, 1, 1]
#         rnd_noise = torch.rand(1)[0] * ramp_fn(self.ramp) * self.rnd_noise
#
#         contrast_low = 1. - (1. - self.contrast_low) * ramp_fn(self.ramp)
#         contrast_high = 1. + (self.contrast_high - 1.) * ramp_fn(self.ramp)
#
#         contrast_params = [contrast_low, contrast_high]
#
#         rnd_sat = torch.rand(1)[0] * ramp_fn(self.ramp) * self.rnd_sat
#
#         if self.apply_many_crops:
#             selected_attack = 'crop'
#             # print(selected_attack + ' from crop list')
#
#         elif self.apply_required_attacks:
#             selected_attack = random.choice(self.required_attack_list)
#             # print(selected_attack + ' from required attack list')
#
#         else:
#             selected_attack = random.choice(['blur', 'noise','contrast','saturation','jpeg', 'resize', 'translate', 'rotate', 'perspective_noise', 'gaussian_noise', 'light_distortion', 'eye_protection', 'gradient_gray'])
#             # selected_attack = random.choice(
#             #     ['blur', 'noise', 'contrast', 'saturation', 'jpeg', 'resize'])
#             # print(selected_attack + ' from all attack list')
#
#         if selected_attack == 'blur':
#             # blur
#             x = (x + 1.) / 2.
#             N_blur = 7
#             f = random_blur_kernel(probs=[.25, .25], N_blur=N_blur, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.],
#                                         wmin_line=3).to(device)
#             x = F.conv2d(x, f, bias=None, padding=int((N_blur - 1) / 2))
#             x = torch.clamp(x, 0, 1)
#             x = (x * 2.) - 1.
#
#         elif selected_attack == 'noise':
#             # noise
#             x = (x + 1.) / 2.
#             noise = torch.normal(mean=0, std=rnd_noise, size=x.size(), dtype=torch.float32).to(device)
#             x = x + noise
#             x = torch.clamp(x, 0, 1)
#             x = (x * 2.) - 1.
#
#         elif selected_attack == 'contrast':
#             # contrast & brightness
#             x = (x + 1.) / 2.
#             contrast_scale = torch.Tensor(x.size()[0]).uniform_(contrast_params[0], contrast_params[1])
#             contrast_scale = contrast_scale.reshape(x.size()[0], 1, 1, 1).to(device)
#             x = x * contrast_scale
#             x = torch.clamp(x, 0, 1)
#
#             x = (x * 2.) - 1.
#
#
#
#         elif selected_attack == 'saturation':
#             # saturation
#             x = (x + 1.) / 2.
#             sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1, 3, 1, 1).to(device)
#             encoded_image_lum = torch.mean(x * sat_weight, dim=1).unsqueeze_(1)
#             x = (1 - rnd_sat) * x + rnd_sat * encoded_image_lum
#             x = torch.clamp(x, 0, 1)
#             x = (x * 2.) - 1.
#
#
#
#         elif selected_attack == 'jpeg':
#             # augly implementation:
#             x = (x + 1.) / 2.
#             x = self.imgnet_normalize(x)
#             jpeg_quality = int(np.random.uniform(40. , 100.))
#             x = self.jpeg_compress(x, jpeg_quality)
#             x = self.imgnet_unnormalize(x)
#             x = (x * 2.) - 1.
#
#
#
#         elif selected_attack == 'resize':
#
#             ## resize 2:
#             x = (x + 1.) / 2.
#             resize_scale_range = (0.5, 1.5)
#             new_w = int(np.random.uniform(*resize_scale_range) * x.size()[3])
#             new_h = int(np.random.uniform(*resize_scale_range) * x.size()[2])
#             x = functional.resize(x, (new_h, new_w), interpolation=functional.InterpolationMode('bilinear'))
#             x = (x * 2.) - 1.
#
# #######################################################################################################################################
#         if selected_attack == 'blur':
#             x = (x + 1.) / 2.
#             N_blur = 7
#             f = random_blur_kernel(probs=[.25, .25], N_blur=N_blur, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.],
#                                    wmin_line=3).to(device)
#             x = F.conv2d(x, f, bias=None, padding=int((N_blur - 1) / 2))
#             x = torch.clamp(x, 0, 1)
#             x = (x * 2.) - 1.
#
#         elif selected_attack == 'translate':
#             x = (x + 1.) / 2.
#             severity = 1
#             device = x.device
#             ct = random.uniform(0, 10)
#             d = ct * severity
#             image_tensor = x
#             dx = random.uniform(-d, d)
#             dy = random.uniform(-d, d)
#             B, C, H, W = image_tensor.shape
#             matrix = _compute_translation_matrix_single(dx, dy, device)
#             if matrix.ndim == 2:
#                 matrix = matrix.unsqueeze(0)
#             matrix = matrix.expand(B, -1, -1)
#             warped = warp_affine(image_tensor, matrix, dsize=(H, W), padding_mode='zeros')
#             x = torch.clamp(warped, 0, 1)
#             x = (x * 2.) - 1.
#
#         elif selected_attack == 'rotate':
#             x = (x + 1.) / 2.
#             severity = 1
#             cr = random.uniform(0, 10)
#             max_deg = cr * severity
#             device = x.device
#             img_t = x
#             B, C, H, W = img_t.shape
#             angle = random.uniform(-max_deg, max_deg)
#             angle_t = torch.tensor([angle], device=device, dtype=torch.float32)
#             center = torch.tensor([[W / 2 - 1, H / 2 - 1]], device=device, dtype=torch.float32)
#             scale = torch.tensor([[1.0, 1.0]], device=device, dtype=torch.float32)
#             matrix = get_rotation_matrix2d(center, angle_t, scale)
#             if matrix.shape[0] != B:
#                 matrix = matrix.expand(B, -1, -1)
#             warped = warp_affine(img_t, matrix, dsize=(H, W), padding_mode='zeros')
#             x = torch.clamp(warped, 0, 1)
#             x = (x * 2.) - 1.
#
#         elif selected_attack == 'perspective_noise':
#             x = (x + 1.) / 2.
#             severity = 1
#             d_values = [8, 16, 32, 48, 64]
#             d = d_values[severity - 1]
#             B, C, H, W = x.shape
#             device = x.device
#             points_src = torch.tensor([[
#                 [0., 0.], [W - 1., 0.], [W - 1., H - 1.], [0., H - 1.]
#             ]], device=device).repeat(B, 1, 1)
#             points_dst = torch.zeros_like(points_src)
#             for i in range(B):
#                 tl_x, tl_y = random.uniform(-d, d), random.uniform(-d, d)
#                 bl_x, bl_y = random.uniform(-d, d), random.uniform(-d, d)
#                 tr_x, tr_y = random.uniform(-d, d), random.uniform(-d, d)
#                 br_x, br_y = random.uniform(-d, d), random.uniform(-d, d)
#                 points_dst[i] = torch.tensor([
#                     [tl_x, tl_y], [tr_x + W - 1, tr_y], [br_x + W - 1, br_y + H - 1], [bl_x, bl_y + H - 1],
#                 ], device=device)
#             M = kornia.geometry.get_perspective_transform(points_src, points_dst)
#             warped = kornia.geometry.transform.warp_perspective(x.float(), M, dsize=(H, W))
#             x = torch.clamp(warped, 0, 1)
#             x = (x * 2.) - 1.
#
#         elif selected_attack == 'gaussian_noise':
#             x = (x + 1.) / 2.
#             severity = 1
#             c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
#             x = x + torch.randn_like(x) * c
#             x = torch.clamp(x, 0, 1)
#             x = (x * 2.) - 1.
#
#         elif selected_attack == 'light_distortion':
#             x = (x + 1.) / 2.
#             severity = 1
#             embed_image = x
#             B, C, H, W = embed_image.shape
#             a = 0.9 - severity * 0.1
#             b = 1.9 + severity * 0.1
#             c_choice = random.randint(0, 1)
#             if c_choice == 0:
#                 mask_2d = np.ones((H, W), dtype=np.float32)
#                 direction = np.random.randint(1, 5)
#                 for i in range(H):
#                     mask_2d[i, :] = -((b - a) / (W - 1)) * (i - W) + a
#                 O = np.rot90(mask_2d, k=direction - 1)
#                 if O.shape != (H, W): O = O.T
#                 mask = torch.from_numpy(O.copy()).to(x.device).view(1, 1, H, W).expand(B, C, H, W)
#             else:
#                 mask = torch.ones((B, C, H, W), device=x.device)
#                 for b_idx in range(B):
#                     x_center, y_center = np.random.randint(0, H), np.random.randint(0, W)
#                     max_len = max(np.sqrt(x_center ** 2 + y_center ** 2), np.sqrt((x_center - H) ** 2 + y_center ** 2),
#                                   np.sqrt(x_center ** 2 + (y_center - W) ** 2),
#                                   np.sqrt((x_center - H) ** 2 + (y_center - W) ** 2))
#                     y_grid, x_grid = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device),
#                                                     indexing='ij')
#                     dist = torch.sqrt((y_grid - x_center) ** 2 + (x_grid - y_center) ** 2)
#                     val = dist / max_len * (a - b) + b
#                     mask[b_idx] = val
#             x = torch.clamp(embed_image * mask, 0, 1)
#             x = (x * 2.) - 1.
#
#         elif selected_attack == 'eye_protection':
#             x = (x + 1.) / 2.
#             severity = 1
#             blue_reduce = 0.8 - 0.1 * severity
#             warm_boost = 1.05 + 0.05 * severity
#             gamma = 1.0 - 0.05 * severity
#             brightness = 0.9 + 0.1 * severity
#             multiplier = torch.tensor([warm_boost * brightness, brightness, blue_reduce * brightness],
#                                       device=x.device).view(1, 3, 1, 1)
#             x = x * multiplier
#             x = torch.clamp(x, 0, 1)
#             x = torch.pow(x, gamma)
#             x = (x * 2.) - 1.
#
#         elif selected_attack == 'gradient_gray':
#             x = (x + 1.) / 2.
#             perspective_sev, gray_sev, dark_factor = 3, 1.0, 0.6
#             d = [2, 5, 10, 15, 20][perspective_sev - 1]
#             B, C, H, W = x.shape
#             device = x.device
#             points_src = torch.tensor([[0., 0.], [W - 1., 0.], [W - 1., H - 1.], [0., H - 1.]],
#                                       device=device).unsqueeze(0).repeat(B, 1, 1)
#             points_dst = torch.zeros_like(points_src)
#             for i in range(B):
#                 tl_x, tl_y = random.uniform(-d, d), random.uniform(-d, d)
#                 tr_x, tr_y = random.uniform(-d, d), random.uniform(-d, d)
#                 br_x, br_y = random.uniform(-d, d), random.uniform(-d, d)
#                 bl_x, bl_y = random.uniform(-d, d), random.uniform(-d, d)
#                 points_dst[i] = torch.tensor(
#                     [[tl_x, tl_y], [tr_x + W - 1, tr_y], [br_x + W - 1, br_y + H - 1], [bl_x, bl_y + H - 1]],
#                     device=device)
#             M = kornia.geometry.get_perspective_transform(points_src, points_dst)
#             x = kornia.geometry.transform.warp_perspective(x, M, dsize=(H, W))
#             gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]).repeat(1, 3, 1, 1)
#             y_g, x_g = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
#             coords = torch.stack((x_g, y_g), dim=-1).float().unsqueeze(0).repeat(B, 1, 1, 1)
#             alpha_list = []
#             max_dist = (H ** 2 + W ** 2) ** 0.5
#             for i in range(B):
#                 pair = random.choice([(0, 1), (1, 2), (2, 3), (3, 0)])
#                 c1, c2 = points_dst[i, pair[0]], points_dst[i, pair[1]]
#                 d1, d2 = torch.norm(coords[i] - c1.view(1, 1, 2), dim=-1), torch.norm(coords[i] - c2.view(1, 1, 2),
#                                                                                       dim=-1)
#                 alpha = torch.clamp(1.0 - torch.minimum(d1, d2) / max_dist, 0., 1.)
#                 alpha_list.append((alpha ** 3) * gray_sev)
#             alpha_mask = torch.stack(alpha_list).unsqueeze(1).clamp(0., 1.)
#             x = alpha_mask * gray + (1.0 - alpha_mask) * x
#             x = x * (1.0 - alpha_mask * (1.0 - dark_factor))
#             x = torch.clamp(x, 0, 1)
#             x = (x * 2.) - 1.
#
#         return x
#
#
#
#
#
# def random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line):
#     N = N_blur
#     coords = torch.from_numpy(np.stack(np.meshgrid(range(N_blur), range(N_blur), indexing='ij'), axis=-1)) - (0.5 * (N-1)) # （7,7,2)
#     manhat = torch.sum(torch.abs(coords), dim=-1)   # (7, 7)
#
#     # nothing, default
#     vals_nothing = (manhat < 0.5).float()           # (7, 7)
#
#     # gauss
#     sig_gauss = torch.rand(1)[0] * (sigrange_gauss[1] - sigrange_gauss[0]) + sigrange_gauss[0]
#     vals_gauss = torch.exp(-torch.sum(coords ** 2, dim=-1) /2. / sig_gauss ** 2)
#
#     # line
#     theta = torch.rand(1)[0] * 2.* np.pi
#     v = torch.FloatTensor([torch.cos(theta), torch.sin(theta)]) # (2)
#     dists = torch.sum(coords * v, dim=-1)                       # (7, 7)
#
#     sig_line = torch.rand(1)[0] * (sigrange_line[1] - sigrange_line[0]) + sigrange_line[0]
#     w_line = torch.rand(1)[0] * (0.5 * (N-1) + 0.1 - wmin_line) + wmin_line
#
#     vals_line = torch.exp(-dists ** 2 / 2. / sig_line ** 2) * (manhat < w_line) # (7, 7)
#
#     t = torch.rand(1)[0]
#     vals = vals_nothing
#     if t < (probs[0] + probs[1]):
#         vals = vals_line
#     else:
#         vals = vals
#     if t < probs[0]:
#         vals = vals_gauss
#     else:
#         vals = vals
#
#     v = vals / torch.sum(vals)
#     z = torch.zeros_like(v)
#     f = torch.stack([v,z,z, z,v,z, z,z,v], dim=0).reshape([3, 3, N, N])
#     return f
#
# def get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size):
#     rnd_hue = torch.FloatTensor(batch_size, 3, 1, 1).uniform_(-rnd_hue, rnd_hue)
#     rnd_brightness = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(-rnd_bri, rnd_bri)
#     return rnd_hue + rnd_brightness
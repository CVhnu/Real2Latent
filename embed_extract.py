import os
import glob
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from torchvision import transforms
from einops import rearrange
from ldm.util import instantiate_from_config
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(config_path, ckpt_path, force_message_len=None):

    print(f"Loading model from {ckpt_path}...")


    config = OmegaConf.load(config_path)


    if force_message_len is not None:
        try:
            old_len = config.model.params.decoder_config.params.message_len
            if old_len != force_message_len:
                print(f"[*] Overriding config message_len: {old_len} -> {force_message_len}")
                config.model.params.decoder_config.params.message_len = force_message_len
        except Exception as e:
            print(f"[!] Warning: Could not override message_len in config structure: {e}")


    model = instantiate_from_config(config.model)


    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']


    misses, ignores = model.load_state_dict(state_dict, strict=False)
    if misses:
        print(f"Missed keys: {len(misses)} (Ensure your checkpoint matches the message_len)")

    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model, config


def process_image_paths(folder_path):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    image_paths.sort()
    return image_paths


def run_embed(args):
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)


    target_len = args.message_len
    if args.message:
        target_len = len(args.message)


    model, config = load_model(args.config, args.weight, force_message_len=target_len)


    try:
        model_msg_len = config.model.params.decoder_config.params.message_len
    except:
        model_msg_len = target_len  # Fallback


    if args.message:
        if len(args.message) != model_msg_len:
            print(f"[!] Warning: Input message length ({len(args.message)}) matches your request, "
                  f"but make sure your checkpoint supports it.")

        message_list = [int(x) for x in args.message]
        message_tensor = torch.tensor(message_list, dtype=torch.float).to(device)
        message_tensor = 2. * message_tensor - 1.
        message_str = args.message
    else:

        print(f"Generating random {model_msg_len}-bit message...")
        message_tensor = torch.randint(0, 2, (model_msg_len,), device=device).float()
        message_str = "".join([str(int(x)) for x in message_tensor.tolist()])
        message_tensor = 2. * message_tensor - 1.


    msg_save_path = os.path.join(args.outdir, "watermark_msg.txt")
    with open(msg_save_path, "w") as f:
        f.write(message_str)
    print(f"[*] Watermark Message ({len(message_str)} bits): {message_str}")
    print(f"[*] Message saved to: {msg_save_path}")

    message_tensor = message_tensor.unsqueeze(0)


    image_paths = process_image_paths(args.image_folder)
    print(f"Found {len(image_paths)} images.")

    transform = transforms.Compose([
        transforms.Resize((args.H, args.W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    batch_size = args.batch_size
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    print("Starting embedding...")
    with torch.no_grad(), precision_scope("cuda"):
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            imgs = [transform(Image.open(p).convert('RGB')) for p in batch_paths]
            batch_img = torch.stack(imgs).to(device)

            batch_msg = message_tensor.repeat(len(imgs), 1)

            z = model.encode_first_stage(batch_img)
            post_z = model.ae.post_quant_conv(1. / model.scale_factor * z)

            _, watermarked_imgs = model(post_z, batch_img, batch_msg)

            watermarked_imgs = torch.clamp((watermarked_imgs + 1.0) / 2.0, min=0.0, max=1.0)

            for idx, w_tensor in enumerate(watermarked_imgs):
                name_no_ext = os.path.splitext(os.path.basename(batch_paths[idx]))[0]
                w_np = 255. * rearrange(w_tensor.cpu().numpy(), 'c h w -> h w c')
                Image.fromarray(w_np.astype(np.uint8)).save(os.path.join(args.outdir, f"{name_no_ext}.png"))

    print("Embedding finished.")





def run_embed(args):
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    target_len = args.message_len
    if args.message:
        target_len = len(args.message)

    model, config = load_model(args.config, args.weight, force_message_len=target_len)

    try:
        model_msg_len = config.model.params.decoder_config.params.message_len
    except:
        model_msg_len = target_len  # Fallback

    if args.message:
        if len(args.message) != model_msg_len:
            print(f"[!] Warning: Input message length ({len(args.message)}) matches your request, "
                  f"but make sure your checkpoint supports it.")

        message_list = [int(x) for x in args.message]
        message_tensor = torch.tensor(message_list, dtype=torch.float).to(device)
        message_tensor = 2. * message_tensor - 1.
        message_str = args.message
    else:
        print(f"Generating random {model_msg_len}-bit message...")
        message_tensor = torch.randint(0, 2, (model_msg_len,), device=device).float()
        message_str = "".join([str(int(x)) for x in message_tensor.tolist()])
        message_tensor = 2. * message_tensor - 1.

    msg_save_path = os.path.join(args.outdir, "watermark_msg.txt")
    with open(msg_save_path, "w") as f:
        f.write(message_str)
    print(f"[*] Watermark Message ({len(message_str)} bits): {message_str}")
    print(f"[*] Message saved to: {msg_save_path}")

    message_tensor = message_tensor.unsqueeze(0)

    image_paths = process_image_paths(args.image_folder)
    print(f"Found {len(image_paths)} images.")

    transform = transforms.Compose([
        transforms.Resize((args.H, args.W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    batch_size = args.batch_size
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    print("Starting embedding (using Residual Addition)...")
    with torch.no_grad(), precision_scope("cuda"):
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            orig_imgs = [Image.open(p).convert('RGB') for p in batch_paths]
            imgs = [transform(img) for img in orig_imgs]
            batch_img = torch.stack(imgs).to(device)
            batch_msg = message_tensor.repeat(len(imgs), 1)
            z = model.encode_first_stage(batch_img)
            post_z = model.ae.post_quant_conv(1. / model.scale_factor * z)
            _, watermarked_imgs = model(post_z, batch_img, batch_msg)
            res = watermarked_imgs.clamp(-1.0, 1.0) - batch_img
            for idx in range(len(batch_paths)):
                orig_img = orig_imgs[idx]
                w, h = orig_img.size
                single_res = res[idx:idx + 1]
                single_res_up = F.interpolate(single_res, size=(h, w), mode='bilinear', align_corners=False)
                single_res_up_np = single_res_up[0].permute(1, 2, 0).cpu().numpy()
                orig_img_np = np.array(orig_img) / 127.5 - 1.0
                stego_np = single_res_up_np + orig_img_np
                stego_uint8 = np.clip(stego_np, -1.0, 1.0) * 127.5 + 127.5
                stego_uint8 = stego_uint8.astype(np.uint8)
                name_no_ext = os.path.splitext(os.path.basename(batch_paths[idx]))[0]
                Image.fromarray(stego_uint8).save(os.path.join(args.outdir, f"{name_no_ext}.png"))
    print("Embedding finished.")



def run_extract(args):
    target_len = len(args.message)
    model, _ = load_model(args.config, args.weight, force_message_len=target_len)

    target_msg = [int(x) for x in args.message]
    target_tensor = torch.tensor(target_msg, dtype=torch.float).to(device)
    target_bool = target_tensor > 0.5

    image_paths = process_image_paths(args.image_folder)
    print(f"Found {len(image_paths)} images. Calculating accuracy for {target_len}-bit message.")

    transform = transforms.Compose([
        transforms.Resize((args.H, args.W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    accuracies = []
    print("-" * 50)
    print(f"{'Filename':<30} | {'Bit Accuracy'}")
    print("-" * 50)

    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            decoded_logits = model.decoder(img_tensor)
            extracted_bool = decoded_logits > 0

            correct_bits = ~(torch.logical_xor(extracted_bool, target_bool.unsqueeze(0)))
            acc = torch.mean(correct_bits.float()).item()

            accuracies.append(acc)
            print(f"{os.path.basename(path):<30} | {acc:.4f}")

    avg_acc = np.mean(accuracies) if accuracies else 0.0
    print("-" * 50)
    print(f"Overall Average Bit Accuracy: {avg_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Watermark Tool")
    parser.add_argument('-c', "--config", default='configs/inference.yaml')
    parser.add_argument('-w', "--weight", required=True, help="Model checkpoint")

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Embed parser
    p_emb = subparsers.add_parser('embed')
    p_emb.add_argument("--image_folder", required=True)
    p_emb.add_argument("--outdir", required=True)
    p_emb.add_argument("--message", type=str, default="", help="Input your 48-bit string here")
    p_emb.add_argument("--message_len", type=int, default=32, help="Default length if message is not provided")
    p_emb.add_argument("--batch_size", type=int, default=4)
    p_emb.add_argument("--H", type=int, default=512)
    p_emb.add_argument("--W", type=int, default=512)
    p_emb.add_argument("--seed", type=int, default=42)
    p_emb.add_argument("--precision", default="autocast")

    # Extract parser
    p_ext = subparsers.add_parser('extract')
    p_ext.add_argument("--image_folder", required=True)
    p_ext.add_argument("--message", type=str, required=True, help="The original 48-bit string")
    p_ext.add_argument("--H", type=int, default=512)
    p_ext.add_argument("--W", type=int, default=512)

    args = parser.parse_args()

    if args.command == 'embed':
        run_embed(args)
    elif args.command == 'extract':
        run_extract(args)

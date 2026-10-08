# Real2Latent: Misalignment-Robust Transfer of Physical Degradations for Screen-Shooting-Resistant Watermarking

## Environment

We have tested our code with the following environment:

* Python 3.8.20
* PyTorch 2.0.1
* Torchvision 0.15.2
* CUDA 11.7

### Embedding and Extraction

To generate watermarked images and extract the embedded messages, run:

```bash
CUDA_VISIBLE_DEVICES=1 python embed_extract.py \
    -w /***/***/***/***/checkpoints/r2l.ckpt \
    embed \
    --image_folder /******/ \
    --outdir /******/ \
    --message "****************************"

CUDA_VISIBLE_DEVICES=1 python embed_extract.py \
    -w /***/***/***/***/checkpoints/r2l.ckpt \
    extract \
    --image_folder /******/ \
    --message "****************************"
```

* `--image_folder` specifies the input image directory.
* `--outdir` specifies the output directory for generated watermarked images.
* `--message` specifies the binary watermark message.

## Training

To train the modified decoder from scratch, run:

```bash
python train.py \
    --message_len 48 \
    --config configs/train.yaml \
    --batch_size 8 \
    --learning_rate 0.00006
```

Here, `--message_len` specifies the watermark length in bits, while `--config` specifies the training configuration file.

We will keep this code continuously updated.

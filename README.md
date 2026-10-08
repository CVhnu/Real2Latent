# Real2Latent: Misalignment-Robust Transfer of Physical Degradations for Screen-Shooting-Resistant Watermarking

![Image 1](https://github.com/CVhnu/sepwater/blob/97651a3fa16365ca4dba0fd1b8307e093e799bb4/EDDD63931DD6647AD4F6321DD9E89398.png)
![Image 2](https://github.com/CVhnu/sepwater/blob/cbd71d3a4b265640fd10dcb65fe17f7c8dee6d98/26C4C8E6A5B57D70391B5A39B3171BFF.png)

## Environment

We have tested our code with the following environment:

* Python 3.8.20
* PyTorch 2.0.1
* Torchvision 0.15.2
* CUDA 11.7


# git clone this repository
git clone https://github.com/CVhnu/Real2Latent.git

# statement
we will integrated the accompanying test dataset into the download link. You can directly obtain the trained weight and test files through the link above. We've also optimized the naming conventions for functions and files, further improving code readability. We will continue to update this link with updates on subsequent iterations！


## The released model can be downloaded at
[Download](https://drive.google.com/drive/folders/1NC5q8BsrUntZ6y_5o267GXnaowoNXp90?usp=sharing)

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

We will keep this code continuously updated！！！

# Citation
If you find our repo useful for your research, please cite us:

# License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only. Any commercial use should get formal permission first.

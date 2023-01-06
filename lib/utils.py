#!/usr/bin/env python
# coding: utf-8
"""
Created on 22/02/2021 12:04

@author: Matheus Silva
"""
# ### built-in deps
import pathlib

# ### external deps
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from skimage import exposure, img_as_ubyte
from skimage.exposure import match_histograms


def normalize(image, reverse=False):
    """
    """
    # ### y=((x-min)/(max-min))*255
    image = np.array(image)
    norm_img  = ((image - np.min(image)))/(np.max(image) - np.min(image))
    return norm_img * 255


def image_loader_new(image_name, loader, device):
    """
    """
    # ### load images and apply transform
    image = Image.open(image_name)    

    pix = img_as_ubyte(image)

    print(f'{str(image_name)}\t({image.getextrema()[0]}, {image.getextrema()[1]})\t({pix.dtype}, {pix.min()}, {pix.max()})')

    pix_rgb = np.stack((pix,)*3, axis=-1).transpose((0, 1, 2))

    PIL_image = Image.fromarray(np.uint8(pix_rgb)).convert('RGB')
    image2 = loader(PIL_image).unsqueeze(0)

    return image2.to(device, torch.float)


def image_loader(image_name, loader, device):
    """
    """
    # ### load images and apply transform
    image = Image.open(image_name)    

    image_ = np.array(image)

    pix = normalize(image)

    print(f'{str(image_name)}\t({image.getextrema()[0]}, {image.getextrema()[1]})\t({image_.dtype}, {image_.min()}, {image_.max()})\t({pix.dtype}, {pix.min()}, {pix.max()})')

    pix_rgb = np.stack((pix,)*3, axis=-1).transpose((0, 1, 2))

    PIL_image = Image.fromarray(np.uint8(pix_rgb)).convert('RGB')
    image2 = loader(PIL_image).unsqueeze(0)

    return image2.to(device, torch.float)
    

def save_samples(unloader, tensor, title=None, save_path="./", original=False):
    """
    """
    image = tensor.cpu().clone().detach() 
    image = image.squeeze(0)    
    image = unloader(image)

    if original:
        pathlib.Path(f'{save_path}/original').mkdir(exist_ok=True)
        image.save(f'{save_path}/original/{title}.tif')
    else:
        pathlib.Path(f'{save_path}/results').mkdir(exist_ok=True)
        image.save(f'{save_path}/results/{title}.tif')


def get_input_optimizer(input_img, lr_):
    """
    """
    # ### this line to show that input is a parameter that requires a gradient
    optimizer = optim.Adam([input_img.requires_grad_()], lr=lr_)
    
    return optimizer

def get_matched_image(output, reference):
    """
    """
    # ### get histogram specification
    matched = match_histograms(output, reference, multichannel=True)

    return matched
    
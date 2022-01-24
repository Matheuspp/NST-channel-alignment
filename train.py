#!/usr/bin/env python
# coding: utf-8
"""
Created on 22/02/2021 12:04

@author: Matheus Silva
"""
# ### built-in deps
import os
import pathlib
import glob
import random
import copy
import time

# ### external deps
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

# ### local deps
from lib.nst_network import ContentLoss, Normalization, StyleLoss
from lib.utils import normalize, save_samples, image_loader, get_input_optimizer

# ### Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# ### GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ### desired size of the output image
imsize = 256 if torch.cuda.is_available() else 128  # use small size if no gpu

# ### loaders
loader = transforms.Compose([transforms.Resize(imsize),  # scale imported image
                             transforms.ToTensor()])  # transform it into a torch tensor
unloader = transforms.ToPILImage()  # reconvert into PIL image

# ### VGG config
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,content_layers, style_layers):

    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)            
            # ### The in-place version doesn't play very nicely with the ContentLoss
            # ### and StyleLoss we insert below. So we replace with out-of-place
            # ### ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # ### add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # ### add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # ### now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=500,
                       style_weight=1000000, content_weight=1, lr=0.001):

    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean,
                                          normalization_std, style_img, content_img,
                                          content_layers=content_layers_default,
                                          style_layers=style_layers_default)


    optimizer = get_input_optimizer(input_img, lr_=lr)
    style_loss_list = []
    content_loss_list = []

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # ### correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
                style_loss_list.append(style_score.item())
                content_loss_list.append(content_score.item())

            return style_score + content_score

        optimizer.step(closure)

    # ### a last correction...
    input_img.data.clamp_(0, 1)
    
    return input_img, style_loss_list, content_loss_list

def train(data_path, save_folder, grid_params=None):

    files = sorted(list(pathlib.Path(data_path).glob('*.tif')))
    batches = [files[i:i+5] for i in range(0, len(files), 5)]
    pathlib.Path(f'{save_folder}').mkdir(parents=True, exist_ok=True)

    for steps in grid_params[0]:
        for lr in grid_params[1]:
            pathlib.Path(f'{save_folder}/sample_result_{steps}_{lr}').mkdir(exist_ok=True)
            for imgs in batches:  
                content_img = image_loader(imgs[1], loader, device) # green
                skip = 0
                for channel in imgs:
                    if skip != 1:
                        style_img = image_loader(channel, loader, device)
                        skip += 1 
                    else:
                        skip += 1 
                        continue       

                    input_img = content_img.clone().detach()
                
                    output, style_loss, content_loss = run_style_transfer(cnn, cnn_normalization_mean, 
                                                       cnn_normalization_std, content_img, style_img,
                                                       input_img, num_steps=steps, lr=lr)

                    txt_path = f'{save_folder}/sample_result_{steps}_{lr}/loss_{steps}_{lr}_{channel.stem}.txt'                              
                    with open(txt_path, 'w') as fp:
                        for i in range(len(style_loss)):
                            fp.write('Style Loss: {:4f} Content Loss: {:4f}\n'.format(style_loss[i], content_loss[i]))
                    
                    # ### save generated styles
                    save_samples(unloader, output, title=f'output_{channel.stem}',
                                 save_path=f'{save_folder}/sample_result_{steps}_{lr}')
                    # ### save original stacked images
                    if steps == grid_params[0][0]:
                        save_samples(unloader, style_img, title=f'{channel.stem}',
                                     save_path=f'{save_folder}', original=True)

                save_samples(unloader, content_img, title=f'output_{imgs[1].stem}',
                       save_path=f'{save_folder}/sample_result_{steps}_{lr}')

    # ### saving original ones with normalization
    for imgs in batches:
        for channel in imgs:
            std_image = image_loader(channel, loader, device)
            save_samples(unloader, std_image, title=f'{channel.stem}',
                         save_path=f'{save_folder}', original=True)   

if __name__ == '__main__':
    lr_list = np.arange(0.001, 0.01, 0.001)
    iter_list = [100, 2000, 3000]
    save_folder = './experiments/search'
    train('./data/soybean', save_folder, grid_params=[iter_list, lr_list])
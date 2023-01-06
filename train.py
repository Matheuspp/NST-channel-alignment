#!/usr/bin/env python
# coding: utf-8
"""
Created on 22/02/2021 12:04

@author: Matheus Silva
"""
# ### built-in deps
import os
import sys
import argparse as ap
import pathlib
import glob
import random
import copy
import time
import datetime
import shutil

# ### external deps
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

# ### local deps
from lib.nst_network import ContentLoss, Normalization, StyleLoss
from lib.utils import normalize, save_samples, image_loader, image_loader_new, get_input_optimizer

# Define o tamanho das fontes para os plots. Matplotlib.
plt.rcParams.update({'font.size': 22})

# Global variables
# ----------------
IMG_LOADER_NEW = True

# ### Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# ### GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ### desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

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

    # just in order to have an iterable access to or list of content/syle losses
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


    print('Building the style transfer model...')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean,
                                          normalization_std, style_img, content_img,
                                          content_layers=content_layers_default,
                                          style_layers=style_layers_default)


    optimizer = get_input_optimizer(input_img, lr_=lr)

    output_img_list = []

    # Lista contendo as listas de perdas.
    # Cada sub lista armazena as perdas para um número de épocas.
    style_loss_list_list = []
    content_loss_list_list = []
    loss_list_list = []

    for s in num_steps:
        style_loss_list_list.append([])
        content_loss_list_list.append([])
        loss_list_list.append([])
    

    print('Optimizing...')
    run = [0]
    # while run[0] <= num_steps:
    # while run[0] <= max(num_steps): # Para N = 5: [0, 1, 2, 3, 4, 5]
    while run[0] < max(num_steps): # Para N = 5: [0, 1, 2, 3, 4, 5]

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

            # Incrementa. Com 1 iteração, run = 1. Com N iterações run = N
            run[0] = run[0] + 1

            for i, s in enumerate(num_steps): # [1000, 2000, 3000]
                if run[0] % 100 == 0 and run[0] <= s:
                    print("run {}: Style Loss: {:4f} Content Loss: {:4f}".format(run[0], style_score.item(), content_score.item()))

                    # style_loss_list_list[i].append(style_score.item())
                    # content_loss_list_list[i].append(content_score.item())
                
                style_loss_list_list[i].append(style_score.item())
                content_loss_list_list[i].append(content_score.item())

                loss_list_list[i].append(loss.item())

            if run[0] in num_steps:
                print(f'Saving output image for \'run\' =  {run[0]}: {style_score}, {content_score}')
                output_img_list.append(input_img.detach().clone())


            # TEST
            #  - Verificar se estamos treinando STEPS ou STEPS+1
            ### print(f'====> TEST <==== {run[0]}, {style_score}, {content_score}')
            return style_score + content_score

        
        optimizer.step(closure)

    # ### a last correction...
    ### input_img.data.clamp_(0, 1)
    for output in output_img_list:
        output.data.clamp_(0, 1)

    # return input_img, style_loss_list, content_loss_list
    return output_img_list, style_loss_list_list, content_loss_list_list, loss_list_list


def train(data_path, save_folder, params=None):
    """
    """

    files = sorted(list(pathlib.Path(data_path).glob('*.tif')))
    batches = [files[i:i+5] for i in range(0, len(files), 5)]

    steps_b = [params[0][i:i+4] for i in range(0, len(params[0]), 4)]
    lr_b = [params[1][i:i+4] for i in range(0, len(params[1]), 4)]

    pathlib.Path(f'{save_folder}{os.path.sep}nst').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{save_folder}{os.path.sep}original').mkdir(parents=True, exist_ok=True)

    # Pasta para armazenar os arquivos de loss (.txt) e os arquivos de gráficos de treinamento (.png) 
    pathlib.Path(f'{save_folder}{os.path.sep}_reports').mkdir(parents=True, exist_ok=True)

    print('\nSTART - Training...')
    print(f'>>>{datetime.datetime.now()}')

    # Relatório CSV
    with open(f'{save_folder}/train_results_new.csv', 'w') as r_fp:
        r_fp.write('Image,Iterations,Learning rate,Style loss,Content loss,Loss\n')

    for imgs, steps, lr in zip(batches, steps_b, lr_b):
        ### pathlib.Path(f'{save_folder}').mkdir(exist_ok=True)

        # Converte de uint16 para uint8 sem modificar o contraste
        # ----->
        if IMG_LOADER_NEW:
            content_img = image_loader_new(imgs[1], loader, device) # green
        else:
            content_img = image_loader(imgs[1], loader, device) # green
        # <-----

        skip = 0
        # for channel in imgs:
        for ch_img, ch_param in zip([0, 2, 3, 4], [0, 1, 2, 3]):
            print(f'Processing {imgs[1].stem} (content) and {imgs[ch_img].stem} (style) with {steps[ch_param]} steps and lr of {lr[ch_param]:0.3f}...')

            # Avoid the green channel (content channel), in the image list.
            #  - Green channel does not exists in param list
            # ----->
            if IMG_LOADER_NEW:
                style_img = image_loader_new(imgs[ch_img], loader, device)
            else:
                style_img = image_loader(imgs[ch_img], loader, device)
            # <-----
        
            input_img = content_img.clone().detach()
        
            output, style_loss, content_loss, loss = run_style_transfer(cnn, cnn_normalization_mean, 
                                                cnn_normalization_std, content_img, style_img,
                                                input_img, num_steps=[steps[ch_param]], lr=lr[ch_param])

            # TEMP
            # print(type(style_loss))
            # print(style_loss)
            # print(type(content_loss))
            # print(content_loss)

            txt_path = f'{save_folder}{os.path.sep}_reports/losses_{imgs[ch_img].stem}_{steps[ch_param]}_{lr[ch_param]:0.3f}.txt'                              
            with open(txt_path, 'w') as fp:
                for i in range(len(style_loss[0])):
                    fp.write('Style Loss: {:4f} Content Loss: {:4f} Loss: {:4f}\n'.format(style_loss[0][i], content_loss[0][i], loss[0][i]))

            with open(f'{save_folder}/train_results_new.csv', 'a') as r_fp:
                r_fp.write(f'{imgs[ch_img].stem},{steps[ch_param]},{lr[ch_param]:0.3f},{style_loss[0][-1]},{content_loss[0][-1]},{loss[0][-1]}\n')

            style_loss_ = style_loss[0]
            content_loss_ = content_loss[0]
            loss_ = loss[0]

            # Plota a o gráfico de treinamento - loss.
            plt.figure(figsize=(10, 8))
            plt.plot(style_loss_, label='Style loss')
            plt.plot(content_loss_, label='Content loss')
            plt.plot(loss_, label='Loss')
            plt.legend()
            plt.savefig(f'{save_folder}{os.path.sep}_reports/training_report_{imgs[ch_img].stem}_{steps[ch_param]}_{lr[ch_param]:0.3f}.png', bbox_inches='tight')
            plt.close()

            # Plota a o gráfico de treinamento - loss.
            plt.figure(figsize=(10, 8))
            plt.plot(style_loss_, label='Style loss')
            plt.legend()
            plt.savefig(f'{save_folder}{os.path.sep}_reports/training_report_style_{imgs[ch_img].stem}_{steps[ch_param]}_{lr[ch_param]:0.3f}.png', bbox_inches='tight')
            plt.close()

            # Plota a o gráfico de treinamento - loss.
            plt.figure(figsize=(10, 8))
            plt.plot(content_loss_, label='Content loss')
            plt.legend()
            plt.savefig(f'{save_folder}{os.path.sep}_reports/training_report_content_{imgs[ch_img].stem}_{steps[ch_param]}_{lr[ch_param]:0.3f}.png', bbox_inches='tight')
            plt.close()

            # Plota a o gráfico de treinamento - loss.
            plt.figure(figsize=(10, 8))
            plt.plot(loss_, label='Loss')
            plt.legend()
            plt.savefig(f'{save_folder}{os.path.sep}_reports/training_report_loss_{imgs[ch_img].stem}_{steps[ch_param]}_{lr[ch_param]:0.3f}.png', bbox_inches='tight')
            plt.close()

            
            # Salva as imagen geradas NST
            # -----
            # ### save generated styles (B, R, RE, NIR)
            save_samples(unloader, output[0], title=f'output_{imgs[ch_img].stem}', save_path=f'{save_folder}{os.path.sep}nst')
            # Save the CONTENT image (Green)...
            save_samples(unloader, content_img, title=f'output_{imgs[1].stem}', save_path=f'{save_folder}{os.path.sep}nst')

            # Salva as imagens originais também...
            # -----
            save_samples(unloader, style_img, title=f'output_{imgs[ch_img].stem}', save_path=f'{save_folder}{os.path.sep}original')
            # Save the CONTENT image (Green)...
            save_samples(unloader, content_img, title=f'output_{imgs[1].stem}', save_path=f'{save_folder}{os.path.sep}original')

    print('\nnFINISH - Training...')
    print(f'<<<{datetime.datetime.now()}')


def search(data_path, save_folder, grid_params=None):
    """
    """

    files = sorted(list(pathlib.Path(data_path).glob('*.tif')))
    batches = [files[i:i+5] for i in range(0, len(files), 5)]
    pathlib.Path(f'{save_folder}').mkdir(parents=True, exist_ok=True)

    # Lista com os números de passos. Ex.: [1000, 2000, 3000]
    steps_list = grid_params[0]

    print('\nSTART - Searching...')
    print(f'>>>{datetime.datetime.now()}')

    # for steps in grid_params[0]:
    for lr in grid_params[1]:
        print(f'\nLearning rate: {lr}')

        for steps in steps_list:
            pathlib.Path(f'{save_folder}/sample_result_{steps}_{lr:0.3f}').mkdir(exist_ok=True)

        for imgs in batches:  

            if IMG_LOADER_NEW:
                # Converte de uint16 para uint8 (sem modificar o contraste)
                content_img = image_loader_new(imgs[1], loader, device) # green
            else:
                # Converte de uint16 para uint8 (ajusta o contraste)
                content_img = image_loader(imgs[1], loader, device) # green
                
            
            skip = 0
            for channel in imgs:
                if skip != 1:
                    # >>> 
                    if IMG_LOADER_NEW:
                        # Converte de uint16 para uint8 sem modificar o contraste
                        style_img = image_loader_new(channel, loader, device)
                    else:
                        style_img = image_loader(channel, loader, device)
                    # <<<

                    skip += 1 

                else:
                    skip += 1 
                    continue       

                input_img = content_img.clone().detach()
            
                # output, style_loss e conten_loss agora são listas. Um elemento para cada número de passos.
                output, style_loss, content_loss = run_style_transfer(cnn, cnn_normalization_mean, 
                                                    cnn_normalization_std, content_img, style_img,
                                                    input_img, num_steps=steps_list, lr=lr)

                for s, steps in enumerate(steps_list):
                    txt_path = f'{save_folder}/sample_result_{steps}_{lr:0.3f}/loss_{steps}_{lr:0.3f}_{channel.stem}.txt'                              
                    with open(txt_path, 'w') as fp:
                        for i in range(len(style_loss[s])):
                            fp.write('Style Loss: {:4f} Content Loss: {:4f}\n'.format(style_loss[s][i], content_loss[s][i]))
                    
                    # ### save generated styles
                    save_samples(unloader, output[s], title=f'output_{channel.stem}', save_path=f'{save_folder}/sample_result_{steps}_{lr:0.3f}')

                    # Save the CONTENT IMAGE (Green)...
                    save_samples(unloader, content_img, title=f'output_{imgs[1].stem}', save_path=f'{save_folder}/sample_result_{steps}_{lr:0.3f}')

    print('\nFINISH - Searching...')
    print(f'<<<{datetime.datetime.now()}')

    # ### saving original ones with normalization
    for imgs in batches:
        for channel in imgs:
            if IMG_LOADER_NEW:
                std_image = image_loader_new(channel, loader, device)
            else:
                std_image = image_loader(channel, loader, device)

            save_samples(unloader, std_image, title=f'{channel.stem}',
                         save_path=f'{save_folder}', original=True)   
                

if __name__ == '__main__':
    # =========================================================================
    # 1) Executar otimização de hiperparametros:
    #   $ python train.py
    # 
    # Para gerar o arquivo 'results.csv' e aplicar a especificação de histograma 
    #   executar o script 'match.py'
    #   $ python match.py experiments
    #
    # 2) Executar o treinamento com os hiperparametros no arquivo 'results.csv':
    #   $ python train_2.py results.csv
    # 
    # Para aplicar a especificação de histograma executar o script 'match.py'
    #   $ python match.py experiments_train
    # =========================================================================

    print('\n\n' + ('=' * 40))
    print('(\'train.py\' - Starting new experiment...')
    print('--> IMG_LOADER_NEW: ' + str(IMG_LOADER_NEW))

    if len(sys.argv) > 1:
        print('\nTraining...')

        # Caminho do arquivo de parametros
        ### filename = args.params
        filename = sys.argv[1]
        print(filename)

        try:
            hyper_params = pd.read_csv(filename)
        
        except:
            print(f'ERROR: File {filename} not found! Exiting...')
            print('Done!')
            sys.exit()

        iter_list = hyper_params['Iteration'].to_list()
        lr_list = hyper_params['Learning Rate'].to_list()

        print(type(iter_list))
        print(iter_list)

        print(type(lr_list))
        print(lr_list)

        # Experiment
        # ==========
        save_folder_root = './experiments_train'
        
        ### save_folder = save_folder_root + '/nst'
        save_folder = save_folder_root 

        train('./data/soybean', save_folder, params=[iter_list, lr_list])

        if os.path.exists('./nohup.out'):
            shutil.move('./nohup.out', f'{save_folder_root}/nohup_train.out')

    else:
        print('\nSearching...')

        lr_list = np.arange(0.001, 0.01, 0.001)
        iter_list = [1000, 2000, 3000]

        # Experiment
        # ==========
        save_folder_root = './experiments'

        save_folder = save_folder_root + '/search'

        search('./data/soybean', save_folder, grid_params=[iter_list, lr_list])
        # <<<<<<<<<<<

        if os.path.exists('./nohup.out'):
            shutil.move('./nohup.out', f'{save_folder_root}/nohup.out')


    print('\'train.py\' - Done!')
#!/usr/bin/env python
# coding: utf-8
"""
Created on 22/02/2021 12:04

@author: Matheus Silva
"""
# ### built-in deps

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


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):

    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, image):
        # ### normalize image
        return (image - self.mean) / self.std

def gram_matrix(input):
    a, b, c, d = input.size() 
    features = input.view(a * b, c * d)  # resize
    # ### computer inner product
    G = torch.mm(features, features.t()) 

    # ### return the gram matrix normalization
    return G.div(a * b * c * d)
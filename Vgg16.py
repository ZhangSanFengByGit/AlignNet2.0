# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 20:04:44 2018

@author: cvpr
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16_bn

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16,self).__init__()
        features = list(vgg16_bn(pretrained=True).features)[:43]
        self.features = nn.ModuleList(features).eval()
        
        
    def forward(self,x):
        
        for ii,model in enumerate(self.features):
            x=model(x)
        
        return x
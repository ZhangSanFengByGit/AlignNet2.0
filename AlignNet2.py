# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 20:05:26 2018

@author: cvpr
"""


import torch.nn as nn
from Vgg16 import Vgg16
from kron import KronEmbed
from ResidualBlock import ResidualBlock
from UpLayer import UpLayer
from ConvLayer import ConvLayer
from collections import namedtuple


class AlignNet2(nn.Module):
    def __init__(self):
        super(AlignNet2,self).__init__()
        
        #self.vgg = Vgg16().eval()
        self.kron = KronEmbed()
        
        self.res1 = ResidualBlock(512)
        self.res2 = ResidualBlock(512)
        self.res3 = ResidualBlock(512)
        
        
        self.up1 = UpLayer(512,256,kernel_size=3,stride=1,upsample=2)
        self.bn1 = nn.BatchNorm2d(256,affine=True)
        
        self.up2 = UpLayer(256,128,kernel_size=3,stride=1,upsample=2)
        self.bn2 = nn.BatchNorm2d(128,affine=True)
        
        self.up3 = UpLayer(128,64,kernel_size=3,stride=1,upsample=2)
        self.bn3 = nn.BatchNorm2d(64,affine=True)
        
        self.up4 = UpLayer(64,32,kernel_size=3,stride=1,upsample=2)
        self.bn4 = nn.BatchNorm2d(32,affine=True)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.final = ConvLayer(32,3,kernel_size=9,stride=1)
        
        
    def forward(self,R,T):
        results = []
        
        
        T_tran1 = self.kron(R,T)
        T_tran1 = self.res1(T_tran1)
        T_tran1 = self.res2(T_tran1)
        T_tran1 = self.res3(T_tran1)
        
        
        T_tran2 = self.kron(R,T_tran1)
        
        T_tran2 = self.relu(self.bn1(self.up1(T_tran2)))
        T_tran2 = self.relu(self.bn2(self.up2(T_tran2)))
        T_tran2 = self.relu(self.bn3(self.up3(T_tran2)))
        T_tran2 = self.relu(self.bn4(self.up4(T_tran2)))
        T_tran2 = self.final(T_tran2)
        
        results.append(T_tran2)
        
        outputs = namedtuple("AlignNetOutputs",['T_tran_image'])
        
        return outputs(*results)
        
        
        
        
        
        
        
        
        
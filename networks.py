#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:43:32 2021

@author: srigowri
"""

import torch
import torch.nn as nn
from torchvision import transforms


class DoubleConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,bias=False),  #same input height and widht
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(out_channel,out_channel,3,1,1,bias=False),  #same input height and widht
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self,in_channel=3, out_channel=3,features= [64,128,256,512],):
        super(Generator,self).__init__()
        self.downward  = nn.ModuleList()
        self.upward = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


        for feature in features:
            self.downward.append(DoubleConv(in_channel,feature))
            in_channel = feature


        for feature in reversed(features):
            self.upward.append(
                nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2)# doubles the height and width

            )
            self.upward.append(DoubleConv(feature*2,feature))

        self.bottleneck = DoubleConv(features[-1],features[-1]*2)

        self.fully_conv = nn.Conv2d(features[0],out_channel,kernel_size=1) #1*1 convolution
        self.tanh = nn.Tanh()

    def forward(self,x):
        
        skip_connect = []
        for down in self.downward:
            x = down(x)   
            skip_connect.append(x)
            x = self.maxpool(x)
        x = self.bottleneck(x)
        

        skip_connect = skip_connect[::-1]

        for idx in range(0,len(self.upward),2):
            x = self.upward[idx](x)

            
            skip_c = skip_connect[idx//2]
            if x.shape!= skip_c.shape:
                x = transforms.functional.resize(x, size = skip_c.shape[2:])

            concat_skip = torch.cat((skip_c,x),dim=1)
            x = self.upward[idx+1](concat_skip)
        
        

        return self.tanh(self.fully_conv(x))


# def test():
#     x = torch.randn((3,3,256,256))

#     model = Generator(in_channel=3,out_channel=3)
#     pred = model(x)

#     print(x.shape)
#     print(pred.shape)
#     assert (pred.shape ==x.shape)
# test()
        

class SingleConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(SingleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,bias=False),  #same input height and widht
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        
        )
    def forward(self,x):
        return self.conv(x)
class Discriminator(nn.Module):
    def __init__(self,in_channel=3,out_channel=1,features= [32,64,128,256]):
        super(Discriminator,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = 2
        self.downward  = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downward.append(SingleConv(in_channel,feature))
            in_channel = feature

        self.final =  nn.Conv2d(256,1,1)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        
        for down in self.downward:
            # print(x.size())
            x = down(x)    
            # print(x.size())        
            x = self.maxpool(x)
            # print(x.size())

        x = self.final(x)
        # print(x.size())
        return self.sigmoid(x)

# def test():
#     x = torch.randn((1,3,256,256))
#     y = torch.randn((1,3,256,256))

#     model = Discriminator(in_channel=3,out_channel=1)
#     pred = model(x)


# test()
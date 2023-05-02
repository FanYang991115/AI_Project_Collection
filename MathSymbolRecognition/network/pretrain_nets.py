'''
Author: Yang Fan
Date: 2022-11-30 10:56:43
LastEditors: Yang Fan
LastEditTime: 2022-12-03 01:46:45
github: https://github.com/FanYang991115
Copyright (c) 2022 by Fan Yang, All Rights Reserved. 
'''


import torch
import torch.nn as nn
import timm


def resnet50():
    model = timm.create_model(
        'resnet50', pretrained=True, in_chans=1, num_classes=10)
    return model


def efficientnet_b4():
    model = timm.create_model(
        'efficientnet_b4', pretrained=True, in_chans=1, num_classes=10)
    return model



def mobilenetv3_rw():
    model = timm.create_model(
        'mobilenetv3_rw', pretrained=True, in_chans=1, num_classes=10)
    return model


class LeNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(True)
 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        self.fc1 = nn.Linear(165888, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
 
        if init_weights:
            self._initialize_weights()
 
    def forward(self, x):
        x = self.conv1(x)  
        x = self.relu(x) 
        x = self.maxpool1(x) 
        x = self.conv2(x) 
        x = self.relu(x)
        x = self.maxpool2(x) 
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x) 
        x = self.relu(x) 
        x = self.fc2(x) 
        x = self.relu(x) 
        x = self.fc3(x)
 
        return x
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


LeNet = LeNet
resnet50 = resnet50
efficientnet_b4 = efficientnet_b4
mobilenetv3_rw = mobilenetv3_rw
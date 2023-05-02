'''
Author: Yang Fan
Date: 2022-11-30 10:50:25
LastEditors: Yang Fan
LastEditTime: 2022-12-02 10:21:13
github: https://github.com/FanYang991115
Copyright (c) 2022 by Fan Yang, All Rights Reserved. 
'''

import torch
import numpy as np
import torch.nn.functional as F
from timm.data import create_transform
from torch.utils.data import Dataset
from PIL import Image


class GetLoader(Dataset):
    def __init__(self, data_train, labels_train, transform, is_train=True):
        self.data = data_train
        self.label = F.one_hot(torch.from_numpy(
            np.array(labels_train).astype(np.int64)))
        self.trans = transform

    def __getitem__(self, index):
        data = Image.fromarray(
            self.data[index].reshape(300, 300).astype('uint8'))
        data = self.trans(data).cuda()
        labels = self.label[index].cuda()
        return data, labels

    def __len__(self):
        return len(self.label)


def build_transform(is_train):
    if is_train:
        transform = create_transform(
            128, is_training=True, mean=(0.4560), std=(0.2240),scale=(0.55,1.0))
    else:
        transform = create_transform(
            128, is_training=False, mean=(0.4560), std=(0.2240))
    return transform


def build_dataset(data_train, labels_train, is_train):
    transform = build_transform(is_train)
    dataset = GetLoader(data_train, labels_train, transform, is_train=True)
    return dataset
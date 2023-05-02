'''
Author: Yang Fan
Date: 2022-11-30 10:54:48
LastEditors: Yang Fan
LastEditTime: 2022-12-03 03:05:53
github: https://github.com/FanYang991115
Copyright (c) 2022 by Fan Yang, All Rights Reserved. 
'''

import argparse, os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import time
import timm.optim.optim_factory as optim_factory
from util.dataset_aug import build_dataset
from network.pretrain_nets import efficientnet_b4, mobilenetv3_rw, resnet50
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

def get_args_parser():
    parser = argparse.ArgumentParser('ML test', add_help=False)
    parser.add_argument('--model', default='mobilenetv3_rw', type=str, metavar='MODEL',
                        help='Name of model to test')
    parser.add_argument('--model_path', default='./model/mobilenetv3_rw', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--data_path', default='./dataset/cleaned_img_data.npy', type=str,
                        help='dataset path')
    parser.add_argument('--label_path', default='./dataset/cleaned_label_data.npy', type=str,
                        help='dataset path')          
    parser.add_argument('--seed', default=0, type=int)
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_test = np.load(args.data_path)
    labels_test = np.load(args.label_path)
    data_test = data_test.transpose(1,0).reshape(-1,1,300,300)

    # load model
    model = globals()[args.model]().cuda()
    model.load_state_dict(torch.load(args.model_path))
    

    # dataloader
    test_dataset = build_dataset(data_test, labels_test,is_train=True)

    test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    drop_last=True,
    )   
    prediction = []
    labels = []
    for img, label in test_dataloader:
        output = model(img)
        pred = output.argmax(dim=1, keepdim=True).flatten()
        label =label.argmax(dim=1, keepdim=True).flatten()
        pred_list = pred.cpu().numpy()
        labe_list = label.cpu().numpy()
        prediction = np.append(prediction,pred_list)
        labels = np.append(labels,labe_list)
    print(prediction.shape)
    print(classification_report(labels,prediction))

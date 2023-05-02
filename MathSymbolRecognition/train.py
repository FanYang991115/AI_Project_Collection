'''
Author: Yang Fan
Date: 2022-11-30 10:54:44
LastEditors: Yang Fan
LastEditTime: 2022-12-03 03:11:42
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
from network.pretrain_nets import efficientnet_b4, mobilenetv3_rw
from sklearn.model_selection import StratifiedKFold

def get_args_parser():
    parser = argparse.ArgumentParser('ML training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=30, type=int)

    # Model parameters
    parser.add_argument('--model', default='efficientnet_b4', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--data_path', default='./dataset/data_train.npy', type=str,
                        help='dataset path')
    parser.add_argument('--label_path', default='./dataset/t_train_corrected.npy', type=str,
                        help='dataset path')                        
    parser.add_argument('--seed', default=0, type=int)


    return parser



if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    KF = StratifiedKFold(n_splits=10)
    device = torch.device(args.device)
    

    data_train = np.load(args.data_path)
    labels_train = np.load(args.label_path)
    data_train = data_train.transpose(1,0).reshape(-1,1,300,300)


    train_dataset = build_dataset(data_train, labels_train,is_train=True)
    train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    drop_last=True,
)

    model = globals()[args.model]().cuda()
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss = nn.CrossEntropyLoss()
    print(optimizer)

    for epoch in range(args.epochs):
        model.train()
        for img, label in train_dataloader:
            l = loss(model(img),label.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

    torch.save(model.state_dict(), f"./model/{args.model}")

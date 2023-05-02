'''
Author: Yang Fan
Date: 2022-11-30 10:54:44
LastEditors: Yang Fan
LastEditTime: 2022-12-03 02:14:29
github: https://github.com/FanYang991115
Copyright (c) 2022 by Fan Yang, All Rights Reserved. 
'''

import argparse, os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import timm.optim.optim_factory as optim_factory
from util.dataset_aug import build_dataset
# from network.LeNet import LeNet
from network.pretrain_nets import resnet50, mobilenetv3_rw, efficientnet_b4
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

def get_args_parser():
    parser = argparse.ArgumentParser('ML training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int)

    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
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

    cudnn.benchmark = True
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    KF = StratifiedKFold(n_splits=10)
    device = torch.device(args.device)
    
    log_writer = SummaryWriter()

    data_train = np.load(args.data_path)
    labels_train = np.load(args.label_path)
    data_train = data_train.transpose(1,0).reshape(-1,1,300,300)


    for train_index, test_index in KF.split(data_train,labels_train):
        X_train, X_test = np.array(data_train)[train_index], np.array(data_train)[test_index]
        y_train, y_test = np.array(labels_train)[train_index], np.array(labels_train)[test_index]
        train_dataset = build_dataset(X_train, y_train,is_train=True)
        train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
    )
        test_dataset = build_dataset(X_test, y_test,is_train=False)
        test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
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
            lr = optimizer.param_groups[0]["lr"]
            log_writer.add_scalar('train_loss', l, epoch)
            log_writer.add_scalar('lr', lr, epoch)

            model.eval()
            correct = 0

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
        
        torch.save(model.state_dict(), f"./model/{args.model}")

import socket
import sys
import gc  # 垃圾回收
import glob
import math
import os
import pickle  # 保存文件
import random
import shutil  # 文件操作
import warnings
from collections import OrderedDict
from pathlib import Path  # 路径
import albumentations as albu  # 图像增强库
import cv2
import matplotlib.pyplot as plt

# import mlflow
import numpy as np
import pandas as pd
import timm  # 开源深度学习库
import torch
import torch.optim as optim
from logzero import logger
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from time import time
from tqdm import tqdm

from config.init import Config
from train_utils.utils import *
from data.utils import *
from data.transform import *
from data.dataset import CVDataSet, CustomDataset
from model.CVNet import CVNet, build_model
from model.U2Net import u2net_full, u2net_lite

warnings.simplefilter("ignore")


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # 获得根目录，数据目录，权值文件目录端到端
    ROOT_DIR, DATA_DIR, OUTPUT_DIR, CP_DIR = Path_init(is_train=True)

    # 设置随机种子
    seed_everything(seed=Config.RANDOM_SATE)

    # 加载数据 三折数据
    data_set = load_data(DATA_DIR, Z_START=Config.Z_START, Z_DIM=Config.Z_DIM)

    # 默认GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"---Using {device} !---")

    # 计时开始
    start_time = time()
    for fold in range(0, 3):
        print(f"====== {fold} ======")

        # 获取数据切片
        print("Get Crop Image...")
        train_images, train_labels, valid_images, valid_labels, valid_masks = get_train_valid_dataset(fold, data_set)
        print(f"--- The train crop_img numbers: {len(train_images)}")
        print(f"--- The valid crop_img numbers: {len(valid_images)}")

        # 创建模型
        # net = CVNet(input_channel=Config.Z_DIM, num_classes=1)
        # net = u2net_full(input_c=Config.Z_DIM, out_ch=1)
        net = build_model(Config)
        net.to(device)
        ckpt_path = None
        if ckpt_path is not None:
            pretrain_weights = torch.load(ckpt_path, map_location='cpu')
            net.load_state_dict({k.replace('module.', ''): v for k, v in pretrain_weights.items()})

        # 损失函数及优化器
        criterion = Criterion(mode=Config.LOSS)
        optimizer = optim.AdamW(net.parameters(), lr=Config.LR, weight_decay=1.0e-02)

        # 数据迭代器创建 完整切片版
        # train_dataset = CVDataSet(
        #     data_set[fold]["train_img"],
        #     get_train_augmentation(),
        #     labels=data_set[fold]["train_label"],
        #     data_type="train",
        #     crop_size=Config.IMG_SIZE,
        # )
        # valid_dataset = CVDataSet(
        #     data_set[fold]["valid_img"],
        #     get_test_augmentation(),
        #     labels=data_set[fold]["valid_label"],
        #     mask=data_set[fold]["valid_mask"],
        #     data_type="valid",
        #     crop_size=Config.IMG_SIZE,
        # )
        #
        # trainloader = DataLoader(
        #     train_dataset,
        #     batch_size=Config.BATCH_SIZE,
        #     pin_memory=True,
        #     shuffle=True,
        #     drop_last=True,
        #     num_workers=Config.NUM_WORKERS // 2
        # )
        # validloader = DataLoader(
        #     valid_dataset,
        #     batch_size=1,
        #     pin_memory=True,
        #     num_workers=Config.NUM_WORKERS // 2
        # )

        # 数据迭代器创建 重叠切片版
        train_dataset = CustomDataset(
            images=train_images,
            transform=get_train_augmentation(),
            labels=train_labels,
            data_type="train",
        )
        valid_dataset = CustomDataset(
            images=valid_images,
            transform=get_test_augmentation(),
            labels=valid_labels,
            mask=valid_masks,
            data_type="valid",
        )
        trainloader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            num_workers=Config.NUM_WORKERS // 2
        )
        validloader = DataLoader(
            valid_dataset,
            batch_size=1,
            pin_memory=True,
            num_workers=Config.NUM_WORKERS // 2
        )

        # 记录日志，以及保存文件
        early_stopping = EarlyStopping(
            patience=Config.PATIENCE, verbose=True, fold=fold, CP_DIR=CP_DIR, NB=Config.NB, HOST=Config.HOST
        )

        # 学习率策略
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            epochs=Config.EPOCH,
            steps_per_epoch=len(trainloader),
            max_lr=Config.MAX_LR,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=1.0e3,
            final_div_factor=1.0e3,
        )

        # 每一个epoch的验证集精确度，学习率
        val_metrics = []
        learning_rates = []
        for epoch in range(Config.EPOCH):
            # 训练一个epoch
            train_one_epoch(epoch, trainloader, validloader, net, criterion, optimizer, scheduler, early_stopping,
                val_metrics, learning_rates, device)

            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        # 保存验证集
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(learning_rates)
        ax2 = ax1.twinx()
        ax2.plot(val_metrics)
        fig.savefig('./result/Auc.png')

        del net, validloader, trainloader, train_dataset, valid_dataset
        torch.cuda.empty_cache()
        gc.collect()
    end_time = time()
    print(f"---The train total time: {end_time - start_time}")
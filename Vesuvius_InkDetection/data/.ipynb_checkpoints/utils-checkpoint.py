import math
import cv2
import gc
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from config.init import Config
from train_utils.distributed_utils import init_distributed_mode, dist, cleanup, reduce_value, is_main_process


def resize(img, DOWNSAMPLING=Config.DOWNSAMPLING, is_img=True):
    if DOWNSAMPLING != 1.:
        size = int(img.shape[1] * DOWNSAMPLING), int(img.shape[0] * DOWNSAMPLING)

        if not is_img:
            img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, size)
    return img

def load_mask(DATA_DIR, index):
    img = cv2.imread(f"{DATA_DIR}/train/{index}/mask.png", 0)
    img = resize(img, is_img=False)

    pad0 = (Config.IMG_SIZE - img.shape[0] % Config.IMG_SIZE)
    pad1 = (Config.IMG_SIZE - img.shape[1] % Config.IMG_SIZE)

    img = np.pad(img, [(0, pad0), (0, pad1)], constant_values=0)

    return img

def load_labels(DATA_DIR, index):
    img = cv2.imread(f"{DATA_DIR}/train/{index}/inklabels.png", 0)
    img = resize(img, is_img=False)

    pad0 = (Config.IMG_SIZE - img.shape[0] % Config.IMG_SIZE)
    pad1 = (Config.IMG_SIZE - img.shape[1] % Config.IMG_SIZE)

    img = np.pad(img, [(0, pad0), (0, pad1)], constant_values=0)

    return img

def load_volume(DATA_DIR, index, Z_START, Z_DIM):
    # A more memory-efficient volune loader
    fnames = [f"{DATA_DIR}/train/{index}/surface_volume/{i:02}.tif"
             for i in range(Z_START, Z_START + Z_DIM)]

    batch_size = 8
    fname_batches = [fnames[i :i + batch_size] for i in range(0, len(fnames), batch_size)]
    volumes = []
    for fname_batch in fname_batches:
        z_slices = []

        if is_main_process():
            fname_batch = tqdm(fname_batch)

        for fname in fname_batch:
            img = cv2.imread(fname, 0)
            img = resize(img)

            pad0 = (Config.IMG_SIZE - img.shape[0] % Config.IMG_SIZE)
            pad1 = (Config.IMG_SIZE - img.shape[1] % Config.IMG_SIZE)
            img = np.pad(img, [(0, pad0), (0, pad1)], constant_values=0)

            z_slices.append(img)
        volumes.append(np.stack(z_slices))
        del z_slices

    return np.concatenate(volumes, axis=0)

def load_sample(DATA_DIR, index, Z_START, Z_DIM):
    if is_main_process():
        print(f"Loading 'train/{index}'...")
    gc.collect()

    return load_volume(DATA_DIR, index, Z_START, Z_DIM), load_mask(DATA_DIR, index), load_labels(DATA_DIR, index)

def load_data(DATA_DIR, Z_START, Z_DIM):
    # img1 = []
    # for i in tqdm(range(65)):
    #     img1.append(cv2.imread(str(DATA_DIR / f"train/1/surface_volume/{i:02}.tif"), 0))
    #
    # img2 = []
    # for i in tqdm(range(65)):
    #     img2.append(cv2.imread(str(DATA_DIR / f"train/2/surface_volume/{i:02}.tif"), 0))
    #
    # img3 = []
    # for i in tqdm(range(65)):
    #     img3.append(cv2.imread(str(DATA_DIR / f"train/3/surface_volume/{i:02}.tif"), 0))
    #
    # img1 = np.stack(img1)  # (65, 8181, 6330)
    # img2 = np.stack(img2)
    # img3 = np.stack(img3)
    #
    # # ink01文件
    # img1_label = cv2.imread(str(DATA_DIR / f"train/1/inklabels.png"), 0)  # (8181, 6330)
    # img2_label = cv2.imread(str(DATA_DIR / f"train/2/inklabels.png"), 0)
    # img3_label = cv2.imread(str(DATA_DIR / f"train/3/inklabels.png"), 0)
    #
    # # mask：是否包含数据
    # img1_mask = cv2.imread(str(DATA_DIR / f"train/1/mask.png"), 0)  # (8181, 6330)
    # img2_mask = cv2.imread(str(DATA_DIR / f"train/2/mask.png"), 0)
    # img3_mask = cv2.imread(str(DATA_DIR / f"train/3/mask.png"), 0)

    img1, img1_mask, img1_label = load_sample(DATA_DIR, index=1, Z_START=Z_START, Z_DIM=Z_DIM)
    img2, img2_mask, img2_label = load_sample(DATA_DIR, index=2, Z_START=Z_START, Z_DIM=Z_DIM)
    img3, img3_mask, img3_label = load_sample(DATA_DIR, index=3, Z_START=Z_START, Z_DIM=Z_DIM)

    data_set = []
    data_set.append(
        {
            "train_img": [img1, img2],
            "train_label": [img1_label, img2_label],
            "valid_img": [img3],
            "valid_label": [img3_label],
            "valid_mask": [img3_mask],
        }
    )

    data_set.append(
        {
            "train_img": [img1, img3],
            "train_label": [img1_label, img3_label],
            "valid_img": [img2],
            "valid_label": [img2_label],
            "valid_mask": [img2_mask],
        }
    )

    data_set.append(
        {
            "train_img": [img2, img3],
            "train_label": [img2_label, img3_label],
            "valid_img": [img1],
            "valid_label": [img1_label],
            "valid_mask": [img1_mask],
        }
    )

    return data_set

def get_train_valid_dataset(fold, data_set):
    train_images = []
    train_labels = []

    valid_images = []
    valid_labels = []
    valid_masks = []
    for fragment_id in tqdm(range(0, 3)):
        if fragment_id == 2:
            image, label, mask = data_set[fold]['valid_img'][0], data_set[fold]['valid_label'][0], data_set[fold]['valid_mask'][0]
        else:
            image, label = data_set[fold]['train_img'][fragment_id], data_set[fold]['train_label'][fragment_id]

        x1_list = list(range(0, image.shape[2] - Config.IMG_SIZE + 1, Config.STRIDE))
        y1_list = list(range(0, image.shape[1] - Config.IMG_SIZE + 1, Config.STRIDE))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + Config.IMG_SIZE
                x2 = x1 + Config.IMG_SIZE

                if fragment_id == 2:
                    valid_images.append(image[:, y1:y2, x1:x2])
                    valid_labels.append(label[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2])
                else:
                    train_images.append(image[:, y1:y2, x1:x2])
                    train_labels.append(label[y1:y2, x1:x2])

    return train_images, train_labels, valid_images, valid_labels, valid_masks
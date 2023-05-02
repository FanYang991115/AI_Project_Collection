import numpy as np
import cv2
import torch
import gc  # 垃圾回收
import math
import glob
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path  # 路径
from model.CVNet import CVNet, build_model
from data.dataset import CVDataSet
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from data.transform import *
from train_utils.utils import *
from config.init import Config


def predict(test_data_dir, CP_DIR, device):
    # 加载测试集数据
    print("--- The test data loading! ---")
    test_img = []
    for i in tqdm(range(Config.Z_START, Config.Z_START + Config.Z_DIM)):
        test_img.append(
            cv2.imread(str(test_data_dir / f"surface_volume/{i:02}.tif"), 0)
        )
    test_img = np.stack(test_img)

    # 加载mask
    test_mask = cv2.imread(str(test_data_dir / "mask.png"), 0)
    print("--- The test data loading over! ---")
    print('\n')

    # 加载三个fold模型文件
    nets = []
    for fold in range(3):
        # net = CVNet(input_channel=Config.Z_DIM, num_classes=1)
        # net = u2net_full(input_c=Config.Z_DIM, out_ch=1)
        # net = build_model(Config, weight=None)
        # net.to(device)
        ckpt_path = CP_DIR / f"{Config.HOST}_{Config.NB}_checkpoint_{fold}.pt"
        # if ckpt_path is not None:
        #     pretrain_weights = torch.load(ckpt_path, map_location='cpu')
        #     net.load_state_dict({k.replace('module.', ''): v for k, v in pretrain_weights.items()})
        net = torch.load(ckpt_path)
        nets.append(net)

    # 创建测试数据集
    test_dataset = CVDataSet(
        [test_img], get_test_augmentation(), data_type="test", crop_size=Config.IMG_SIZE
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=Config.NUM_WORKERS,
    )

    print(f"--- The test {str(test_data_dir).split('/')[-1]} start! ---")
    test_preds = []
    n_iter_val = len(testloader)
    for i, (img, target) in tqdm(enumerate(testloader), total=n_iter_val):
        net.eval()
        with torch.no_grad():
            img, pawpularities = img.to(device).float(), target.to(device).float()

            outputs_all = np.zeros((img.shape[0], img.shape[2], img.shape[3]))

            # 取三个模型预测平均数
            for net in nets:
                outputs = net(img)
                outputs_np = outputs.squeeze().to("cpu").detach().numpy().copy()
                outputs_all += outputs_np / 3

            test_preds.append(outputs_all)
    print(f"--- The test {str(test_data_dir).split('/')[-1]} end! ---")
    print('\n')

    # 清除内存
    del net, testloader, test_dataset
    torch.cuda.empty_cache()
    gc.collect()

    w_count = math.ceil(test_img[0].shape[1] / Config.IMG_SIZE)
    h_count = math.ceil(test_img[0].shape[0] / Config.IMG_SIZE)

    tile_arry = []
    stack_pred = np.vstack(test_preds).reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE)
    for h_i in range(h_count):
        tile_arry.append(stack_pred[h_i * w_count : (h_i + 1) * w_count])

    pred_tile_img = concat_tile(tile_arry)

    pred_tile_img = np.where(
        test_mask > 1,
        pred_tile_img[: test_img[0].shape[0], : test_img[0].shape[1],],
        0,
    )

    return pred_tile_img


if __name__ == '__main__':
    ROOT_DIR = Path("./")
    DATA_DIR = Path("./vesuvius-challenge-ink-detection")
    OUTPUT_DIR = ROOT_DIR / "output"
    CP_DIR = OUTPUT_DIR
    test_root_dir = DATA_DIR / "test/*"

    # 默认GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"---Using {device} !---")

    pred_list = []
    for f in glob.glob(str(test_root_dir)):
        img_id = str(f).split("/")[-1]

        pred_tile_img = predict(Path(f), CP_DIR=CP_DIR, device=device)

        # 保存预测结果图片
        pred_img = np.where(pred_tile_img > Config.TH, 1, 0) * 255
        cv2.imwrite(f'./result/pred{img_id}.png', pred_img)
        print(f"--- The pred {img_id}.png Save! ---")
        print('\n')

        # 提交文件
        # inklabels_rle = rle(pred_tile_img, img_id, thr=0.4)
        inklabels_rle = fast_rle(pred_tile_img, img_id, thr=Config.TH)
        pred_list.append({"Id": img_id, "Predicted": inklabels_rle})

    # 保存提交文件
    pd.DataFrame(pred_list).to_csv("submission.csv", index=False)
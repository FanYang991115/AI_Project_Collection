import torch
import gc
import matplotlib.pyplot as plt
from time import time
from pathlib import Path  # 路径
from torch.utils.data import DataLoader, Dataset
from config.init import Config
from model.CVNet import CVNet, build_model
from data.utils import *
from data.dataset import CVDataSet
from train_utils.utils import *
from data.transform import *
from config.init import Config


ROOT_DIR = Path("./")
DATA_DIR = Path("./vesuvius-challenge-ink-detection")
OUTPUT_DIR = ROOT_DIR / "output"
CP_DIR = OUTPUT_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"---Using {device} !---")

# 以灰度图的形式读取tif数据
data_set = load_data(DATA_DIR, Z_START=Config.Z_START, Z_DIM=Config.Z_DIM)

all_preds = []
all_masks = []
average_auc = 0
start_time = time()
for fold in range(0, 3):
    print(f"====== {fold} ======")

    # net = CVNet(input_channel=Config.Z_DIM, num_classes=1)
    # net = u2net_full(input_c=Config.Z_DIM, out_ch=1)
    # net = build_model(Config, weight=None)
    # net.to(device)
    ckpt_path = CP_DIR / f"{Config.HOST}_{Config.NB}_checkpoint_{fold}.pt"
    # if ckpt_path is not None:
    #     pretrain_weights = torch.load(ckpt_path, map_location='cpu')
    #     net.load_state_dict({k.replace('module.', ''): v for k, v in pretrain_weights.items()})
    net = torch.load(ckpt_path)

    valid_dataset = CVDataSet(
        data_set[fold]["valid_img"],
        get_test_augmentation(),
        labels=data_set[fold]["valid_label"],
        mask=data_set[fold]["valid_mask"],
        data_type="valid",
        crop_size=Config.IMG_SIZE,
    )

    validloader = DataLoader(
        valid_dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=Config.NUM_WORKERS // 2
    )

    auc = vaild(Config, fold, data_set, validloader, net, all_masks, all_preds, device)
    average_auc += auc

    del net, validloader, valid_dataset
    torch.cuda.empty_cache()
    gc.collect()

end_time = time()
print(f"---The valid total time: {end_time - start_time}")
print(f"The auc: {average_auc / 3}")

# 三张图片的展平结果,高度堆叠后展平
flat_preds = np.hstack(all_preds).reshape(-1).astype(np.float64)
flat_masks = (np.hstack(all_masks).reshape(-1) / 255).astype(np.int64)

# 设定不同的阈值，来生成二值图计算三张图片平均f1分数
thr_list = []
for thr in tqdm(np.arange(0.2, 0.6, 0.1)):
    _val_pred = np.where(flat_preds > thr, 1, 0).astype(np.int)
    score = f1_score(flat_masks, _val_pred)
    print(thr, score)
    thr_list.append({"thr": thr, "score": score})
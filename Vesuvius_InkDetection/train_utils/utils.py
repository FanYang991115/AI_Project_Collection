import numpy as np
import cv2
import torch
import math
import pickle  # 保存文件
from torch import nn
from io import StringIO
from collections import OrderedDict
from pathlib import Path  # 路径
from tqdm import tqdm
from logzero import logger
from torch.nn import functional as F
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
from train_utils.distributed_utils import reduce_value, is_main_process
from train_utils.metrics import *


def Path_init(is_train):
    ROOT_DIR = Path("./")
    DATA_DIR = Path("/blue/ruogu.fang/f.yang1/")
    OUTPUT_DIR = ROOT_DIR / "output"
    CP_DIR = OUTPUT_DIR

    return ROOT_DIR, DATA_DIR, OUTPUT_DIR, CP_DIR

# pickle保存数据
def to_pickle(filename, obj):
    with open(filename, mode="wb") as f:
        pickle.dump(obj, f)

# 加载数据
def unpickle(filename):
    with open(filename, mode="rb") as fo:
        p = pickle.load(fo)
    return p


def rle(img, img_id, thr=0.5):
    flat_img = img.flatten()
    flat_img = np.where(flat_img > thr, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix

    predicted = " ".join(map(str, sum(zip(starts_ix, lengths), ())))
    return {"Id": img_id, "Predicted": predicted}


def fast_rle(img, img_id, thr=0.5):
    flat_img = img.flatten()
    flat_img = np.where(flat_img > thr, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    predicted_arr = np.stack([starts_ix, lengths]).T.flatten()
    f = StringIO()
    np.savetxt(f, predicted_arr.reshape(1, -1), delimiter=" ", fmt="%d")
    predicted = f.getvalue().strip()

    return {"Id": img_id, "Predicted": predicted}


# 图像拼接
def concat_tile(im_list_2d):
    im_listh = [cv2.hconcat(im_list_h) for im_list_h in im_list_2d]  # 15 (512, 5632)
    im = cv2.vconcat(im_listh)  # (7680, 5632)
    return im


class Criterion(nn.Module):
    def __init__(self, mode="BCE"):
        super(Criterion, self).__init__()
        self.mode = mode

    def forward(self, inputs, target):
        if self.mode == "BCE":
            losses = [F.binary_cross_entropy_with_logits(inputs[i].reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE), target) for i in range(len(inputs))]

            total_loss = sum(losses)
        elif self.mode == "DICE":
            DiceLoss = smp.losses.DiceLoss(mode='binary')
            losses = [DiceLoss(inputs[i].reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE), target) for i in range(len(inputs))]

            total_loss = sum(losses)
        elif self.mode == "Tversky":
            alpha = 0.5
            beta = 1 - alpha
            TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False, alpha=alpha, beta=beta)
            losses = [TverskyLoss(inputs[i].reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE), target) for i in
                          range(len(inputs))]

            total_loss = sum(losses)
        elif self.mode == "DICE_BCE":
            DiceLoss = smp.losses.DiceLoss(mode='binary')
            losses_D = [DiceLoss(inputs[i].reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE), target) for i in
                          range(len(inputs))]
            losses_B = [
                F.binary_cross_entropy_with_logits(inputs[i].reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE), target) for
                i in range(len(inputs))]

            total_loss = 0.5 * sum(losses_D) + 0.5 * sum(losses_B)
        elif self.mode == "Tversky_BCE":
            alpha = 0.5
            beta = 1 - alpha
            TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False, alpha=alpha, beta=beta)
            losses_T = [TverskyLoss(inputs[i].reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE), target) for i in
                        range(len(inputs))]
            losses_B = [
                F.binary_cross_entropy_with_logits(inputs[i].reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE), target) for
                i in range(len(inputs))]

            total_loss = 0.5 * sum(losses_T) + 0.5 * sum(losses_B)

        return total_loss


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, fold="", CP_DIR=None, NB=None, HOST=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = 0
        self.delta = delta
        self.fold = fold
        self.CP_DIR = CP_DIR
        self.NB = NB
        self.HOST = HOST

    def __call__(self, val_auc, model):

        score = val_auc

        # 第一次进入
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        # 若是小于最佳分数
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        # 若是大于最佳分数,保存权重
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            logger.info(
                f"Validation auc increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ..."
            )

        # torch.save(model.state_dict(), self.CP_DIR / f"{self.HOST}_{self.NB}_checkpoint_{self.fold}.pt")
        torch.save(model, self.CP_DIR / f"{self.HOST}_{self.NB}_checkpoint_{self.fold}.pt")
        self.val_auc_max = val_auc


def train_one_epoch(epoch, trainloader, validloader, net, criterion, optimizer, scheduler, early_stopping,
                    val_metrics, learning_rates, device):
    n_iter = len(trainloader)
    optimizer.zero_grad()
    net.train()

    if is_main_process():
        trainloader = tqdm(trainloader, total=n_iter)

    for i, (img, target, mask) in enumerate(trainloader):
        img, target = img.to(device).float(), target.to(device).float()

        outputs = net(img)

        if isinstance(outputs, list):
            loss = criterion(outputs, target)
        else:
            loss = criterion([outputs], target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print statistics
        loss = reduce_value(loss, average=True)

        if is_main_process():
            trainloader.set_postfix(
                OrderedDict(
                    epoch="{:>10}".format(epoch),
                    loss="{:.4f}".format(loss.item()),
                )
            )

        scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    # 验证过程
    auc_sum = 0
    n_iter_val = len(validloader)

    if is_main_process():
        validloader = tqdm(validloader, total=n_iter_val)

    net.eval()
    count = 0
    for i, (img, target, mask) in enumerate(validloader):
        with torch.no_grad():
            img, label, mask = img.to(device).float(), target.to(device).float(), mask.to(device).float()  # torch.Size([1, 65, 512, 512]) torch.Size([1, 512, 512])
            outputs = net(img)  # torch.Size([1, 1, 512, 512])
            outputs_np = outputs.to("cpu").detach().numpy().copy()
            label = label.to("cpu").detach().numpy().copy()
            mask = mask.to("cpu").detach().numpy().copy()

            pred_img = np.where(mask > 1, outputs_np, 0)

            # 这里是因为若是对应的区域label皆为0，则会报错，缺少二分类的0和1.所以mask之和不为0，但是label之和为0也不参与计算精度
            if sum(mask.reshape(-1)) != 0:
                try:
                    auc = roc_auc_score(
                        label.reshape(-1),
                        pred_img.reshape(-1),
                    )
                    # auc = calc_cv(label, pred_img)
                    count += 1
                    auc_sum += auc
                except ValueError:
                    pass

    # 保存验证精度
    auc = torch.tensor(auc_sum / count, dtype=torch.float32).to(device)

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    auc = reduce_value(auc, average=True)

    if is_main_process():
        logger.info("auc:{:.4f}".format(auc))
        val_metrics.append(auc.to("cpu").detach().numpy().copy())

        # 学习率记录
        lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(lr)

        # 保存权重
        early_stopping(auc, net)


def vaild(Config, fold, data_set, validloader, net, all_masks, all_preds, device):
    val_preds = []
    valid_targets = []
    n_iter_val = len(validloader)

    if is_main_process():
        validloader = tqdm(validloader, total=n_iter_val)

    net.eval()
    for i, (img, target) in enumerate(validloader):
        with torch.no_grad():
            img, pawpularities = img.to(device).float(), target.to(device).float()  # torch.Size([1, 65, 512, 512]) torch.Size([1, 512, 512])
            outputs = net(img)  # torch.Size([1, 1, 512, 512])
            outputs_np = outputs.to("cpu").detach().numpy().copy()

            val_preds.append(outputs_np)
            valid_targets.append(pawpularities.to("cpu").detach().numpy().copy())

    ## 端を切る
    w_count = math.ceil(data_set[fold]["valid_label"][0].shape[1] / Config.IMG_SIZE)
    h_count = math.ceil(data_set[fold]["valid_label"][0].shape[0] / Config.IMG_SIZE)

    tile_arry = []

    #(165, 1, 512, 512) -> (165, 512, 512)
    stack_pred = np.vstack(val_preds).reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE)

    # 由于cellid是按照行划分的第一行为0-wcount
    for h_i in range(h_count):
        # print(len(test_preds[h_i * w_count:(h_i + 1) * w_count]), h_i * w_count, (h_i + 1) * w_count)
        tile_arry.append(stack_pred[h_i * w_count: (h_i + 1) * w_count])

    # tile_arry中保存了hcount个数据,每一个都是一行的图片
    # contact首先对行拼接,变成hcount个行,然后竖直拼接
    pred_tile_img = concat_tile(tile_arry)

    # 由于原图是经过padding到512的倍数的,所以需要取原图尺寸大小
    # 取max大于1区域,这是因为验证集和测试集都有mask,可以约束预测结果mask之外的错误预测,但是训练集需要输出结果所以不用mask
    pred_tile_img = np.where(
        data_set[fold]["valid_mask"][0] > 1,
        pred_tile_img[:data_set[fold]["valid_label"][0].shape[0], :data_set[fold]["valid_label"][0].shape[1]],
        0,
    )  # (7606, 5249) float32 ndarray

    # 计算精度 展平后求取
    auc = roc_auc_score(
        data_set[fold]["valid_label"][0].reshape(-1),  # (7606, 5249) uint8 ndarray
        pred_tile_img.reshape(-1),
    )
    logger.info("auc:{:.4f}".format(auc))
    auc = torch.tensor(auc, dtype=torch.float32).to(device)

    # 保存对应mask和预测结果
    all_masks.append(data_set[fold]["valid_label"][0].reshape(-1))
    all_preds.append(pred_tile_img.reshape(-1))

    del img, target, outputs

    return auc
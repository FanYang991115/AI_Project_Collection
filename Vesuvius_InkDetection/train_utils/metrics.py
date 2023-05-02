import numpy as np
from config.init import Config

def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()
    cfp = preds[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice

# def calc_fbeta(mask, mask_pred):
#     mask = mask.astype(int).flatten()
#     mask_pred = mask_pred.flatten()
#
#     best_th = 0
#     best_dice = 0
#     for th in np.array(range(10, 50 + 1, 5)) / 100:
#         dice = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)
#
#         if dice > best_dice:
#             best_dice = dice
#             best_th = th
#
#     if is_main_process():
#         print(f'best_th: {best_th}, fbeta: {best_dice}')
#
#     return best_dice, best_th


def calc_fbeta(mask, mask_pred):
    mask = mask.astype(int).flatten()
    mask_pred = mask_pred.flatten()

    dice = fbeta_numpy(mask, (mask_pred >= Config.TH).astype(int), beta=0.5)

    return dice


def calc_cv(mask_gt, mask_pred):
    dice = calc_fbeta(mask_gt, mask_pred)

    return dice
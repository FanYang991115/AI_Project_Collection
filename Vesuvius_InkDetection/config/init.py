import multiprocessing


class Config:
    NB = 'UNet'
    HOST = '00_200_12_1_224_20_15_smp'

    BACKBONE = 'efficientnet-b0'
    # BACKBONE = 'se_resnext50_32x4d'

    LOSS = 'BCE'

    RANDOM_SATE = 42  # 设置随机种子

    LR = 1.0e-05  # 学习率
    MAX_LR = 1.0e-05  # 最大学习率

    PATIENCE = 30  # 15个epoch精度没上升就停止训练
    EPOCH = 200
    BATCH_SIZE = 12

    DOWNSAMPLING = 1  # 整张图片下采样率
    IMG_SIZE = 224  # 小切片大小
    STRIDE = IMG_SIZE // 2  # 切片图像步距
    Z_DIM = 20  # Number of slices in the z direction. Max value is 65 - Z_START
    Z_START = 15

    TH = 0.4  # 阈值

    NUM_WORKERS = multiprocessing.cpu_count()
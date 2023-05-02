import albumentations as A  # 图像增强库
from config.init import Config


def get_train_augmentation():
    train_transform = [
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),  # resize
        A.HorizontalFlip(p=0.5),  # 水平翻转
        A.VerticalFlip(p=0.5),  # 竖直翻转
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50]),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(Config.IMG_SIZE * 0.3), max_height=int(Config.IMG_SIZE * 0.3),
                        mask_fill_value=0, p=0.5),
        A.Normalize(mean=[0] * Config.Z_DIM, std=[1] * Config.Z_DIM),  # 标准化
    ]
    return A.Compose(train_transform, additional_targets={'mask1': 'mask'})


def get_test_augmentation():
    train_transform = [
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=[0], std=[1] * Config.Z_DIM),
    ]
    return A.Compose(train_transform, additional_targets={'mask1': 'mask'})
import os
import albumentations as A
import matplotlib.pyplot as plt
from train_utils.utils import *
from data.utils import *
from data.dataset import CVDataSet, CustomDataset


def get_augmentation():
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
    ]
    return A.Compose(train_transform)


figures_dir = '/data/qyl/test/Kaggle/Ink Detection/visual_img'

# 获得根目录，数据目录，权值文件目录端到端
ROOT_DIR, DATA_DIR, OUTPUT_DIR, CP_DIR = Path_init(is_train=True)

# 加载数据 三折数据
data_set = load_data(DATA_DIR, Z_START=Config.Z_START, Z_DIM=Config.Z_DIM)

# 获取数据切片
print("Get Crop Image...")
fold = 0
train_images, train_labels, valid_images, valid_labels, valid_masks = get_train_valid_dataset(fold, data_set)

plot_dataset = CustomDataset(
            images=train_images,
            labels=train_labels,
            data_type="train",
        )

transform = get_augmentation()

plot_count = 0
for i in range(1000):
    image, label, _ = plot_dataset[i]
    data = transform(image=image, mask=label)
    aug_image = data['image']
    aug_label = data['mask']

    if label.sum() == 0:
        continue

    fig, axes = plt.subplots(1, 4, figsize=(15, 8))
    axes[0].imshow(image[..., 0], cmap="gray")
    axes[1].imshow(label, cmap="gray")
    axes[2].imshow(aug_image[..., 0], cmap="gray")
    axes[3].imshow(aug_label, cmap="gray")

    plt.savefig(os.path.join(figures_dir, f'aug_{fold}_{plot_count}.png'))

    plot_count += 1

    if plot_count == 10:
        break
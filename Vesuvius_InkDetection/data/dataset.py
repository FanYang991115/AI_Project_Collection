import math
import numpy as np
from torch.utils.data import DataLoader, Dataset


class CVDataSet(Dataset):
    def __init__(self, imgs, transforms, labels=None, mask=None, data_type=None, crop_size=256):
        self.crop_size = crop_size # 裁剪尺寸
        self.imgs = imgs
        self.transforms = transforms
        self.labels = labels
        self.mask = mask
        self.data_type = data_type

        # 获取图片裁剪后裁剪区域数量
        self.cell_counts = []
        for img in self.imgs:
            # 长宽高度上数量
            cell_count = math.ceil(img.shape[1] / self.crop_size) * math.ceil(
                img.shape[2] / self.crop_size)
            self.cell_counts.append(cell_count)

    def __len__(self):
        data_count = 0
        if self.data_type == "train":
            self.cell_id_maps = {}

            counter = 0
            for img_num, img in enumerate(self.imgs):
                cell_count = math.ceil(img.shape[1] / self.crop_size) * math.ceil(
                    img.shape[2] / self.crop_size
                )
                for cell_id in range(cell_count):
                    h_num = cell_id // math.ceil(
                        self.labels[img_num].shape[1] / self.crop_size
                    )
                    w_num = cell_id - (
                        h_num
                        * math.ceil(self.labels[img_num].shape[1] / self.crop_size)
                    )

                    cropped_img = self.labels[img_num][
                        h_num * self.crop_size : h_num * self.crop_size
                        + self.crop_size,
                        w_num * self.crop_size : w_num * self.crop_size
                        + self.crop_size,
                    ]

                    if cropped_img.sum() == 0:
                        continue

                    data_count += 1

                    self.cell_id_maps[counter] = (img_num, cell_id)
                    counter += 1

        else:
            for img in self.imgs:
                data_count += math.ceil(img.shape[1] / self.crop_size) * math.ceil(
                    img.shape[2] / self.crop_size
                )
        return data_count  # 返回两张图片的总切片数量

    def calc_img_num(self, idx):
        cum_cell_count = 0
        for i, cell_count in enumerate(self.cell_counts):
            cum_cell_count += cell_count
            if idx + 1 <= cum_cell_count:
                return i, idx - (cum_cell_count - cell_count)

    def __getitem__(self, idx):
        if self.data_type == "train":
            img_num, cell_id = self.cell_id_maps[idx]
        else:
            img_num, cell_id = self.calc_img_num(idx)  # 验证集因为切片没有label的区域也参与，所以idx一般等于cellid，或者差datacount。

        target_img = self.imgs[img_num]
        if self.data_type == 'train':
            target_label = self.labels[img_num]
        elif self.data_type == 'valid':
            target_mask = self.mask[img_num]
            target_label = self.labels[img_num]
        else:
            pass

        target_img = np.moveaxis(target_img, 0, 2)

        h_num = cell_id // math.ceil(target_img.shape[1] / self.crop_size)
        w_num = cell_id - (h_num * math.ceil(target_img.shape[1] / self.crop_size))

        # 超出索引不会报错，后面通过数据增强中的resize将不足512的边强行变成512
        cropped_img = target_img[
            h_num * self.crop_size : h_num * self.crop_size + self.crop_size,
            w_num * self.crop_size : w_num * self.crop_size + self.crop_size,
        ]

        if self.data_type == 'train':
            cropped_label = target_label[
                h_num * self.crop_size: h_num * self.crop_size + self.crop_size,
                w_num * self.crop_size: w_num * self.crop_size + self.crop_size,
            ]
            augmented = self.transforms(image=cropped_img, mask=cropped_label)
            img = augmented["image"]
            img = np.moveaxis(img, 2, 0)
            label = augmented["mask"]
            mask = -1
        elif self.data_type == 'valid':
            cropped_label = target_label[
                h_num * self.crop_size : h_num * self.crop_size + self.crop_size,
                w_num * self.crop_size : w_num * self.crop_size + self.crop_size,
            ]
            cropped_mask = target_mask[
                h_num * self.crop_size: h_num * self.crop_size + self.crop_size,
                w_num * self.crop_size: w_num * self.crop_size + self.crop_size,
            ]
            augmented = self.transforms(image=cropped_img, mask=cropped_label, mask1=cropped_mask)
            img = augmented["image"]
            img = np.moveaxis(img, 2, 0)
            label = augmented["mask"]
            mask = augmented['mask1']
        else:
            augmented = self.transforms(image=cropped_img)
            img = augmented["image"]
            img = np.moveaxis(img, 2, 0)
            label = -1
            mask = -1

        return img, label / 255, mask


class CustomDataset(Dataset):
    def __init__(self, images, transform=None, labels=None, mask=None, data_type=None):
        self.images = images
        self.labels = labels
        self.mask = mask
        self.transform = transform
        self.data_type = data_type

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = np.moveaxis(image, 0, 2)

        if self.labels:
            label = self.labels[idx]

        if self.mask:
            mask = self.mask[idx]

        if self.transform:
            if self.data_type == 'train':
                data = self.transform(image=image, mask=label)
                image = data['image']
                image = np.moveaxis(image, 2, 0)
                label = data['mask']
                mask = -1
            elif self.data_type == 'valid':
                data = self.transform(image=image, mask=label, mask1=mask)
                image = data['image']
                image = np.moveaxis(image, 2, 0)
                label = data['mask']
                mask = data['mask1']
            else:
                data = self.transform(image=image)
                image = data['image']
                image = np.moveaxis(image, 2, 0)
                label = -1
                mask = -1
        else:
            if self.data_type == 'train':
                mask = -1
            elif self.data_type == 'valid':
                pass
            else:
                label = -1
                mask = -1

        return image, label / 255, mask
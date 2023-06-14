"""
torchvision 是pytorch 中专门用来处理图像的库,含有四个大类
    torchvision.datasets 加载数据集
    torchvision.models 提供一些已经训练好的模型
    torchvision.transforms 提供图像处理需要的工具, resize, crop, data_augmentation
    torchvision.utils
"""
import os

import skimage.io as io
import torch
import torchvision
from loguru import logger


class CustomDataset(torch.utils.data.Dataset):
    """自定义数据集"""

    def __init__(self, root_dir, names_file, transform=None):
        # 1. Initialize file paths or a list of file names.

        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(f"{self.names_file}  is not exists")
        file = open(self.names_file)
        print(file)
        for f in file:
            self.names_list.append(f)
            self.size += 1
        print(self.names_list)

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        logger.info(f"")
        img_path = os.path.join(self.root_dir, self.names_list[index].split(" ")[0])
        logger.info(f"img_path: {img_path}")
        if not os.path.isfile(img_path):
            logger.warning(f"{img_path} not exists!")
            return None
        image = io.imread(img_path)
        label = int(self.names_list[index].split(" ")[1])
        logger.info(label)
        sample = {"image": image, "label": label}
        # 可以进一步对数据做出处理
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.size


# Download and construct CIFAR-10 dataset.
# train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
#                                              train=True,
#                                              transform=torchvision.transforms.ToTensor(),
#                                              download=True)

# You can then use the prebuilt data loader.
custom_dataset = CustomDataset(
    root_dir="",
    names_file="/media/tx-deepocean/Data/DICOMS/demos/Projects/pytorch-tutorial/txd_learn_notes/test.txt",
)
# print(custom_dataset.__getitem__(0))
train_loader = torch.utils.data.DataLoader(
    dataset=custom_dataset, batch_size=64, shuffle=True
)
print(train_loader)

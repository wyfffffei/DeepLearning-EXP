import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import save_image

from CNN.hw3_main import _readfile

import numpy as np
import model
import os
import torch


# # check in windows
# for d in range(11):
#     os.mkdir(f"./output/{d}")
# for pic in predict.values:
#     file = str(pic[0]).zfill(4)
#     target = str(pic[1])
#     print(f"copy .\\food-11\\testing\\{file}.jpg .\\output\\{target}\\{file}.jpg")
#     os.system(f"copy .\\food-11\\testing\\{file}.jpg .\\output\\{target}\\{file}.jpg")


workspace_dir = './food-11'
# print("Reading data ..")
test_x = _readfile(os.path.join(workspace_dir, "temp"), False)

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 隨機將圖片水平翻轉
    transforms.RandomRotation(15),  # 隨機旋轉圖片
    transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
])

# test_y = ImageFolder(workspace_dir, transform=train_transform)
test_set = model.ImgDataset(test_x, transform=test_transform)
# print("Size of training data = {}".format(train_set.__len__()))
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

# data augmentation
for epo in range(10):
    for num, img in enumerate(test_set):
        print(img.shape)
        save_image(img, "./food-11/temp/img" + str(epo * test_set.__len__() + num) + ".png")


# #  Note
# def test_1(x, y, flag):
#     if flag:
#         return x, y
#     else:
#         return y
# def test_2(x, y, flag):
#     return x, y if flag else x
# print(test_1(1, 2, False))
# print(test_2(1, 2, False))

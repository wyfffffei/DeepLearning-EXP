import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from CNN.main import _readfile

import numpy as np
import model
import os
import torch

# predict = pd.read_csv("./predict.csv")

# for d in range(11):
#     os.mkdir(f"./output/{d}")

# check in windows
# for pic in predict.values:
#     file = str(pic[0]).zfill(4)
#     target = str(pic[1])
#     print(f"copy .\\food-11\\testing\\{file}.jpg .\\output\\{target}\\{file}.jpg")
#     os.system(f"copy .\\food-11\\testing\\{file}.jpg .\\output\\{target}\\{file}.jpg")


workspace_dir = './food-11'
# print("Reading data ..")
# train_x, train_y = _readfile(os.path.join(workspace_dir, "training"), True)
# print("Size of training data = {}".format(len(train_y)))
# test_x = _readfile(os.path.join(workspace_dir, "1"), False)
# print("Size of Testing data = {}".format(len(test_x)), end="\n\n")

# train_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(),  # 隨機將圖片水平翻轉
#     transforms.RandomRotation(15),  # 隨機旋轉圖片
#     transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
# ])
#
# batch_size = 1
# x = torch.rand(8, 2, 2)
# print(x)
#
# train_set = model.ImgDataset(x, transform=train_transform)
# print("Size of training data = {}".format(train_set.__len__()))
# # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
#
# for data in train_set:
#     print(data)


# Note
def test_1(x, y, flag):
    if flag:
        return x, y
    else:
        return y
def test_2(x, y, flag):
    return x, y if flag else x
print(test_1(1, 2, False))
print(test_2(1, 2, False))

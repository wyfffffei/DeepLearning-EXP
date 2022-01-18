import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from CNN.main import _readfile

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


workspace_dir = './output'
print("Reading data ..")
train_x = _readfile(os.path.join(workspace_dir, "1"), False)
print("Size of training data = {}".format(len(train_x)))

batch_size = 32
train_set = model.ImgDataset(train_x)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

for data in train_loader:
    print(data)


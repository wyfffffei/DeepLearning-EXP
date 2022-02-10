from FullConnection import FullConnection
from main import datainit
import torch
import torch.nn.functional as F


net = FullConnection()
# input = torch.randn(54, requires_grad=True)
# out = net(input)
# print(out)

(x_train_data, x_val_data, x_test_data), (y_train, y_val, y_test) = datainit()
x_train_data = torch.from_numpy(x_train_data).float()
y_train = torch.from_numpy(y_train).float()


i=0
for x_train, y_ans in zip(x_train_data, y_train):
    pre_train = net(x_train)
    print(F.binary_cross_entropy(pre_train, y_ans))

    print()
    i+=1
    if i == 10:
        break

import numpy as np
import torch
import torch.nn.functional as F

from main import datainit
from train_eval_utils import test
from FullConnection import FullConnection


# net = FullConnection()
# input = torch.randn(54, requires_grad=True)
# out = net(input)
# print(out)

x_train_data, x_val_data, x_test_data, y_train, y_val, y_test = datainit()
x_test_data = torch.from_numpy(x_test_data).float().cuda()
y_test = torch.from_numpy(y_test).float().cuda()


# i=0
# for x_train, y_ans in zip(x_train_data, y_train):
#     print(y_ans)
#     pre_train = net(x_train)
#     print(pre_train)
#     print(F.binary_cross_entropy(pre_train, y_ans))
#
#     print()
#     i+=1
#     if i == 10:
#         break

# @torch.no_grad()
# def evaluate(x):
#     x = x * 2
#
# x = torch.ones(2, requires_grad=True)
# print(x)
#
# evaluate(x)
# print(x)

model_path = "weights_results/lr_1e-4_wd_1e-4_r10.pt"
print("TEST ACC: {:.4f}%".format(100. * test(model_path, x_test_data, y_test, 0.4)))


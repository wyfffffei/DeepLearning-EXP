import FullConnection
import torch


net = FullConnection()
input = torch.randn(54, requires_grad=True)
out = net(input)
print(out)

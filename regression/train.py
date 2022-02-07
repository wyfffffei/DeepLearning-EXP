import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class FullConnection(nn.Module):
    def __init__(self):
        super(FullConnection, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def dataloader():
    # train_datasets = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
    # train_data = DataLoader(dataset=train_datasets, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
    # test_datasets = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    # test_data = DataLoader(dataset=test_datasets, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
    train_data = torch.randn(784, requires_grad=True)
    test_data = torch.randn(784, requires_grad=True)
    return train_data, test_data


def main():
    device = torch.device("cpu")
    net = FullConnection().to(device)
    learning_rate = 1e-2
    epochs = 100

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss().to(device)
    train_data, test_data = dataloader()
    logger = SummaryWriter("./logs_train")

    # func = FullConnection()
    # input = torch.randn(784, requires_grad=True)
    # out = func(input)
    # print(out)

    for epoch in range(epochs):
        # 一轮训练
        net.train()
        for batch_id, (tra_data, tra_target) in enumerate(train_data):
            tra_data, tra_target = tra_data.to(device), tra_target.to(device)
            pre_train = net(tra_data)
            loss = loss_func(pre_train, tra_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 10 == 0:
                print("Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_id*len(tra_data), len(train_data), 100.*batch_id/len(train_data), loss.item()
                ))
                logger.add_scalar("train_loss", loss.item(), batch_id)

        # 准确度测试
        net.eval()
        test_loss = 0
        accurancy = 0
        with torch.no_grad():
            for tes_data, tes_target in test_data:
                tes_data, tes_target = tes_data.to(device), tes_target.to(device)
                pre_test = net(tes_data)
                test_loss += loss_func(pre_test, tes_target).item()
                accurancy += (pre_test.argmax(1) == tes_target).sum()
        test_loss /= len(test_data)

        print("\nTest set : Average loss: {:.4f}, Accurancy: {}/{}({:.3f}%)".format(
            test_loss, accurancy, len(test_data), 100.*accurancy/len(test_data)
        ))
        logger.add_scalar("test_loss", test_loss, epoch)
        logger.add_scalar("test_accuracy", accurancy, epoch)

    logger.close()
    torch.save(net, "best.pt")


if __name__ == "__main__":
    main()

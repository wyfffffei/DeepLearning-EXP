import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Classifier, ImgDataset
import time


# read the image file
def _readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x, y = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8), None
    if label:
        y = np.zeros(len(image_dir), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (128, 128))
        if label:
            # 图像命名的短横前数字表示类别
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x


def main():
    # 分別读取 training set、validation set、testing set
    workspace_dir = './food-11'
    print("Reading data ..")
    train_x, train_y = _readfile(os.path.join(workspace_dir, "training"), True)
    print("Size of training data = {}".format(len(train_x)))
    val_x, val_y = _readfile(os.path.join(workspace_dir, "validation"), True)
    print("Size of validation data = {}".format(len(val_x)))
    test_x = _readfile(os.path.join(workspace_dir, "testing"), False)
    print("Size of Testing data = {}".format(len(test_x)), end="\n\n")

    # data augmentation
    data_transform = {
        "train": transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),  # 隨機將圖片水平翻轉
            transforms.RandomRotation(15),  # 隨機旋轉圖片
            transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
        ]),
        "val": transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]),
        "test": transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
    }

    device = torch.device("cuda")
    batch_size = 32
    num_epoch = 30
    learning_rate = 1e-3

    # Load the picture matrix
    train_set = ImgDataset(train_x, train_y, data_transform["train"])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_set = ImgDataset(val_x, val_y, data_transform["val"])
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

    model = Classifier().to(device)
    loss = nn.CrossEntropyLoss()  # Loss function -> CrossEntropyLoss (classification)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer -> Adam

    # Training
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
            train_pred = model(data[0].to(device))  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
            batch_loss = loss(train_pred, data[1].to(device))  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
            batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
            optimizer.step()  # 以 optimizer 用 gradient 更新參數值

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].to(device))
                batch_loss = loss(val_pred, data[1].to(device))

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

            # 將結果 print 出來
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f'
                  % (epoch + 1, num_epoch, time.time()-epoch_start_time, train_acc/train_set.__len__(),
                     train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

    # 得到好的参数后，使用 training set 和 validation set 共同训练（资料越多，模型效果越好）
    train_val_x = np.concatenate((train_x, val_x), axis=0)
    train_val_y = np.concatenate((train_y, val_y), axis=0)
    train_val_set = ImgDataset(train_val_x, train_val_y, data_transform["train"])
    train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

    model_best = Classifier().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_best.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0

        model_best.train()
        for i, data in enumerate(train_val_loader):
            optimizer.zero_grad()
            train_pred = model_best(data[0].to(device))
            batch_loss = loss(train_pred, data[1].to(device))
            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        # 將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f'
              % (epoch + 1, num_epoch, time.time()-epoch_start_time,
                 train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))

    # Testing
    test_set = ImgDataset(test_x, transform=data_transform["test"])
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model_best.eval()
    prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model_best(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                prediction.append(y)

    # 將結果寫入 csv 檔
    with open("predict.csv", 'w') as f:
        f.write('Id,Category\n')
        for i, y in enumerate(prediction):
            f.write('{},{}\n'.format(i, y))


if __name__ == "__main__":
    main()


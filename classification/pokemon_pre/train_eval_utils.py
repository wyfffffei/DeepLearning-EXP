import torch
import numpy as np


def train_one_epoch(model, x_train_data, y_train, device, optimizer, loss_fn, logger, epoch):
    model.train()
    total_loss = 0.0
    running_loss = 0.0
    for batch_id, (x_train, y_train_ans) in enumerate(zip(x_train_data, y_train)):
        x_train, y_train_ans = x_train.to(device, dtype=torch.float), y_train_ans.to(device, dtype=torch.float)
        optimizer.zero_grad()

        pre_train = model(x_train)  # 模型预测
        pre_train = pre_train.squeeze()
        loss = loss_fn(pre_train, y_train_ans)  # 计算 Loss 值
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        running_loss += loss.item()
        # 每训练1000条数据，输出这1000次的平均 loss 值
        if (batch_id + 1) % 1000 == 0:
            logger.add_scalar("Train Loss", running_loss / 1000, epoch * len(y_train) + batch_id)
            print("Train Epoch : {} [{}/{} ({:.0f}%)]\t\tLoss : {:.6f}".format(
                epoch + 1, batch_id + 1, len(y_train),
                100. * (batch_id + 1) / len(y_train), np.round(running_loss / 1000, 3)
            ))
            running_loss = 0.0
    print("Train set : Average Loss: {:.6f}, Epoch: {}".format(total_loss / len(y_train), epoch + 1))


@torch.no_grad()
def evaluate_acc(model, x_val_data, y_val, device, loss_fn, logger, epoch, threshold):
    model.eval()
    total_acc = 0.0
    running_acc = 0.0
    total_loss = 0.0
    running_loss = 0.0

    for batch_id, (x_val, y_val_ans) in enumerate(zip(x_val_data, y_val)):
        x_val, y_val_ans = x_val.to(device, dtype=torch.float), y_val_ans.to(device, dtype=torch.float)
        pre_val = model(x_val)
        pre_val = pre_val.squeeze()

        # 计算验证数据集的 ACC 和 LOSS
        if np.abs((pre_val - y_val_ans).cpu()) <= threshold and 0 <= pre_val <= 1:
            running_acc += 1
            total_acc += 1

        loss = loss_fn(pre_val, y_val_ans).item()
        running_loss += loss
        total_loss += loss

        # 每验证1000条数据，输出这1000次的平均 loss 和平均 Accurancy 值
        if (batch_id + 1) % 1000 == 0:
            logger.add_scalar("Validation Loss", running_loss / 1000, epoch * len(y_val) + batch_id)
            logger.add_scalar("Validation Accurancy", running_acc / 1000, epoch * len(y_val) + batch_id)
            running_acc = 0.0
            running_loss = 0.0
    print("Validation set : Average Loss: {:.6f}, Average Accurancy: {:.6f}%\n".format(
        total_loss / len(y_val), 100. * (total_acc / len(y_val))
    ))


@torch.no_grad()
def test(model_path, x_test, y_test, threshold):
    model = torch.load(model_path)
    model.eval()
    right = 0
    # 测试集测试准确率
    for pok, ans in zip(x_test, y_test):
        pre = model(pok)
        pre = pre.squeeze()
        if np.abs((pre - ans).cpu()) <= threshold and 0 <= pre <= 1:
            right += 1
    return right / len(y_test)


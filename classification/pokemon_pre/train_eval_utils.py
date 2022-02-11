import torch
import numpy as np


def train_one_epoch(model, x_train_data, y_train, device, optimizer, loss_fn, logger, epoch):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    for batch_id, (x_train, y_train_ans) in enumerate(zip(x_train_data, y_train)):
        x_train, y_train_ans = x_train.to(device), y_train_ans.to(device)
        pre_train = model(x_train)
        loss = loss_fn(pre_train, y_train_ans)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss = (mean_loss * batch_id + loss.detach()) / (batch_id + 1)  # update mean losses
        # TODO: ¶Ô±È train loss ºÍ validation loss
        logger.add_scalar("Train Loss -- {}".format(batch_id), mean_loss, batch_id)
        if (batch_id + 1) % 1000 == 0:
            print("Train Epoch : {} [{}/{} ({:.0f}%)]\t\tLoss : {:.6f}".format(
                # epoch + 1, batch_id + 1, len(y_train), 100. * (batch_id + 1) / len(y_train), loss.item()
                epoch + 1, batch_id + 1, len(y_train), 100. * (batch_id + 1) / len(y_train), np.round(mean_loss.item(), 3)
            ))
    return mean_loss


@torch.no_grad()
def evaluate_acc(model, x_val_data, y_val, device, loss_fn):
    model.eval()
    acc = []
    loss_list = []
    mean_loss = torch.zeros(1).to(device)
    for batch_id, (x_val, y_val_ans) in enumerate(zip(x_val_data, y_val)):
        x_val, y_val_ans = x_val.to(device), y_val_ans.to(device)
        pre_val = model(x_val)
        acc.append(np.abs(pre_val - y_val_ans))
        loss = loss_fn(pre_val, y_val_ans)
        mean_loss = (mean_loss * batch_id + loss.detach()) / (batch_id + 1)
        loss_list.append(np.round(mean_loss.item(), 3))  # update mean losses
    acc = 1 - np.mean(acc)
    print("Validation set : Average Loss: {:.4f}, Accurancy: {:.4f}%\n".format(
        np.mean(loss_list), 100. * acc
    ))
    return acc


def predict():
    pass


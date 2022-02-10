import numpy as np
import pandas as pd
from FullConnection import FullConnection
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def datainit():
    pokemon_df = pd.read_csv("./pokemon.csv").set_index('#')
    combats_df = pd.read_csv("./combats.csv")

    # 检查数据缺失情况
    # print(pokemon_df.info())
    # print(pokemon_df["Type 2"].value_counts(dropna=False))

    # 填充缺失数据
    pokemon_df["Type 2"].fillna("empty", inplace=True)

    # 检查数据类型
    # print(pokemon_df.dtypes)
    # print('-' * 30)
    # print(combats_df.dtypes)

    # 调整数据类型
    pokemon_df["Type 1"] = pokemon_df["Type 1"].astype("category")
    pokemon_df["Type 2"] = pokemon_df["Type 2"].astype("category")
    pokemon_df["Legendary"] = pokemon_df["Legendary"].astype("int")

    # 种类独热编码
    df_type1_one_hot = pd.get_dummies(pokemon_df["Type 1"])
    df_type2_one_hot = pd.get_dummies(pokemon_df["Type 2"])
    # pd.options.display.max_columns = 30
    pokemon_df = pokemon_df.join(df_type1_one_hot.add(df_type2_one_hot, fill_value=0).astype("int64"))

    # 获取属性字典
    dict_category = dict(enumerate(pokemon_df["Type 2"].cat.categories))
    print(80 * '-')
    print("宝可梦属性字典:")
    print(dict_category)

    # 更新属性的序列值，丢弃无用项
    pokemon_df["Type 1"] = pokemon_df["Type 1"].cat.codes
    pokemon_df["Type 2"] = pokemon_df["Type 2"].cat.codes
    pokemon_df.drop("Name", axis="columns", inplace=True)
    print(80 * '-')
    print("宝可梦属性:")
    print(pokemon_df)

    # 对战结果处理
    combats_df["Winner"] = combats_df.apply(lambda x: 0 if x.Winner == x.First_pokemon else 1, axis="columns")
    print(80 * '-')
    print("训练数据:")
    print(combats_df)

    # 训练、验证、测试数据集划分
    data_num = combats_df.shape[0]
    np.random.seed(66)
    indexes = np.random.permutation(data_num)
    train_indexes = indexes[:int(data_num * 0.1)]
    val_indexes = indexes[int(data_num * 0.9):]
    # train_indexes = indexes[:int(data_num * 0.6)]
    # val_indexes = indexes[int(data_num * 0.6) : int(data_num * 0.8)]
    test_indexes = indexes[int(data_num * 0.8):]  # 暂不使用

    train_data = combats_df.loc[train_indexes]
    val_data = combats_df.loc[val_indexes]
    test_data = combats_df.loc[test_indexes]

    # 归一化
    pokemon_df["Type 1"] = pokemon_df["Type 1"] / 19
    pokemon_df["Type 2"] = pokemon_df["Type 2"] / 19
    mean = pokemon_df.loc[:, 'HP':'Generation'].mean()
    std = pokemon_df.loc[:, 'HP':'Generation'].std()
    pokemon_df.loc[:, 'HP':'Generation'] = (pokemon_df.loc[:, 'HP':'Generation'] - mean) / std
    print(80 * '-')
    print("归一化属性值:")
    print(pokemon_df.head(10))

    # 训练数据转为 Numpy Array 格式
    x_train_index = np.array(train_data.drop("Winner", axis="columns"))
    x_val_index = np.array(val_data.drop("Winner", axis="columns"))
    x_test_index = np.array(test_data.drop("Winner", axis="columns"))
    y_train = np.array(train_data["Winner"])
    y_val = np.array(val_data["Winner"])
    y_test = np.array(test_data["Winner"])

    # 通过 one-hot 编码索引宝可梦属性
    print(80 * '-')
    print("训练数据shape:")
    one_hot_pokemon_df = np.array(pokemon_df.loc[:, "HP":])
    print(one_hot_pokemon_df.shape)  # -> (800, 27) -> 每次输入两个宝可梦 -> reshape(-1, 27 * 2)
    x_train_data = one_hot_pokemon_df[x_train_index - 1].reshape((-1, 27 * 2))
    x_val_data = one_hot_pokemon_df[x_val_index - 1].reshape((-1, 27 * 2))
    x_test_data = one_hot_pokemon_df[x_test_index - 1].reshape((-1, 27 * 2))
    print(x_train_data.shape)  # -> (30000, 54)
    print()
    return ((x_train_data, x_val_data, x_test_data), (y_train, y_val, y_test))


def train(data):
    device = torch.device("cuda")
    net = FullConnection().to(device)
    epochs = 10
    learning_rate = 1e-3

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 优化器 -> Adam
    loss_fn = F.binary_cross_entropy  # 损失函数 -> Binary Cross Entropy
    from torch.utils.tensorboard import SummaryWriter
    logger = SummaryWriter("./logs_train")  # 日志记录 -> Tensorboard

    # 数据读取
    (x_train_data, x_val_data, x_test_data), (y_train, y_val, y_test) = data
    x_train_data = torch.from_numpy(x_train_data).float()
    x_val_data = torch.from_numpy(x_val_data).float()
    y_train = torch.from_numpy(y_train).float()
    y_val = torch.from_numpy(y_val).float()
    
    # 训练开始
    for epoch in range(epochs):
        net.train()
        for batch_id, (x_train, y_train_ans) in enumerate(zip(x_train_data, y_train)):
            x_train, y_train_ans = x_train.to(device), y_train_ans.to(device)
            pre_train = net(x_train)
            loss = loss_fn(pre_train, y_train_ans)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_id + 1) % 500 == 0:
                print("Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss : {:.6f}".format(
                    epoch + 1, batch_id + 1, len(y_train), 100.*(batch_id + 1) / len(y_train), loss.item()
                ))
                logger.add_scalar("train_loss", loss.item(), batch_id)

        net.eval()
        val_loss = 0
        accurancy = 0
        with torch.no_grad():
            for x_val, y_val_ans in zip(x_val_data, y_val):
                x_val, y_val_ans = x_val.to(device), y_val_ans.to(device)
                pre_val = net(x_val)
                val_loss += loss_fn(pre_val, y_val_ans)
                accurancy += 1 if pre_val == y_val_ans else 0
        val_loss /= len(y_val)
        print("Validation set : Average Loss: {:.4f}, Accurancy: {}/{}({:.3f}%)\n".format(
            val_loss, accurancy, len(y_val), 100.*accurancy / len(y_val)
        ))
        logger.add_scalar("Val_Loss", val_loss, epoch)
        logger.add_scalar("Val_Accurancy", accurancy, epoch)

    logger.close()
    torch.save(net, "best.pt")



def main():
    data = datainit()
    print(80 * '-')
    print("训练开始:")
    train(data)
    print("训练结束")
    print(80 * '-')


if __name__ == "__main__":
    main()

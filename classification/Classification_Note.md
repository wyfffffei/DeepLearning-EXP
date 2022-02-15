# Classification_Note

> @ wyfffffei



## Binary Classification

### Logistic Regression

Regression 模型用于对连续数据进行预测，而 logistic regression 模型则是应用于分类问题。

### Sigmoid

由于存在梯度消失的问题，Sigmoid 函数不会应用于隐藏层，但像二分类的逻辑回归模型，该函数往往会被应用于输出层，它的输出介于0和1之间

 ![sigmoid.png](./img/sigmoid.png)

### 二分类交叉熵

在深度学习模型中常见的损失函数有 交叉熵（Cross-Entropy, CE）和均方误差（Mean Squared Error, MSE）。在回归类问题中，常用‘均方误差’作为损失函数，在分类问题中常用‘交叉熵’

 ![loss_selection.png](./img/loss_selection.png)

交叉熵公式：

 ![CE.png](./img/CE.png)

- 二分类：BCE
- 多分类：CCE

### 独热编码（One-Hot Encoding）

类别表示法，对于没有连续关联的类别，如使用0、1、2...方法表示，则存在1比2更接近0的潜在关系；

独热编码将类别以若干个‘0’或1个‘1’表示，如果有N个类别，就以“N-1个0”和“1个1”来表示，确保类别与类别之间是完全独立的

### 应用

> 名称：Pokemon-Challenge
>
> 数据集：<https://www.kaggle.com/terminus7/pokemon-challenge>

#### 数据预处理

```python
pokemon_df = pd.read_csv("./pokemon.csv").set_index('#')
combats_df = pd.read_csv("./combats.csv")

# 数据缺失情况修正
pokemon_df["Type 2"].fillna("empty", inplace=True)

# 数据类型检查修正
pokemon_df["Type 1"] = pokemon_df["Type 1"].astype("category")
pokemon_df["Type 2"] = pokemon_df["Type 2"].astype("category")
pokemon_df["Legendary"] = pokemon_df["Legendary"].astype("int")

# 种类独热编码
df_type1_one_hot = pd.get_dummies(pokemon_df["Type 1"])
df_type2_one_hot = pd.get_dummies(pokemon_df["Type 2"])
pokemon_df = pokemon_df.join(df_type1_one_hot.add(df_type2_one_hot, fill_value=0).astype("int64"))

# 更新属性的序列值，丢弃无用项
pokemon_df["Type 1"] = pokemon_df["Type 1"].cat.codes
pokemon_df["Type 2"] = pokemon_df["Type 2"].cat.codes
pokemon_df.drop("Name", axis="columns", inplace=True)

# 训练数据处理（对战结果）
combats_df["Winner"] = combats_df.apply(lambda x: 0 if x.Winner == x.First_pokemon else 1, axis="columns")

data_num = combats_df.shape[0]
np.random.seed(11)
indexes = np.random.permutation(data_num)
# 训练、验证数据集划分（6 : 2 : 2）
train_indexes = indexes[:int(data_num * 0.6)]
val_indexes = indexes[int(data_num * 0.6) : int(data_num * 0.8)]
test_indexes = indexes[int(data_num * 0.8):]
train_data = combats_df.loc[train_indexes]
val_data = combats_df.loc[val_indexes]
test_data = combats_df.loc[test_indexes]

# 归一化
pokemon_df["Type 1"] = pokemon_df["Type 1"] / 19
pokemon_df["Type 2"] = pokemon_df["Type 2"] / 19
mean = pokemon_df.loc[:, 'HP':'Generation'].mean()
std = pokemon_df.loc[:, 'HP':'Generation'].std()
pokemon_df.loc[:, 'HP':'Generation'] = (pokemon_df.loc[:, 'HP':'Generation'] - mean) / std

# 训练数据转为 Numpy Array 格式
x_train_index = np.array(train_data.drop("Winner", axis="columns"))
x_val_index = np.array(val_data.drop("Winner", axis="columns"))
x_test_index = np.array(test_data.drop("Winner", axis="columns"))
y_train = np.array(train_data["Winner"])
y_val = np.array(val_data["Winner"])
y_test = np.array(test_data["Winner"])

# 使用 one-hot 编码索引宝可梦，数据格式 reshape
one_hot_pokemon_df = np.array(pokemon_df.loc[:, "HP":])
print("原数据: ", end=str(one_hot_pokemon_df.shape)+'\n')  # -> (800, 27) -> 每次输入两个宝可梦 -> reshape(-1, 27 * 2)
x_train_data = one_hot_pokemon_df[x_train_index - 1].reshape((-1, 27 * 2))
x_val_data = one_hot_pokemon_df[x_val_index - 1].reshape((-1, 27 * 2))
x_test_data = one_hot_pokemon_df[x_test_index - 1].reshape((-1, 27 * 2))
print("新数据: ", end=str(x_train_data.shape)+'\n')  # -> (-1, 54)

```

#### 训练（一轮）

```python
def train_one_epoch(model, x_train_data, y_train, device, optimizer, loss_fn, logger, epoch):
    model.train()
    total_loss = 0.0
    running_loss = 0.0
    for batch_id, (x_train, y_train_ans) in enumerate(zip(x_train_data, y_train)):
        x_train, y_train_ans = x_train.to(device), y_train_ans.to(device)
        optimizer.zero_grad()

        pre_train = model(x_train)  # 模型预测
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

```

#### 准确率计算

```python
@torch.no_grad()
def evaluate_acc(model, x_val_data, y_val, device, loss_fn, logger, epoch, threshold):
    model.eval()
    total_acc = 0.0
    running_acc = 0.0
    total_loss = 0.0
    running_loss = 0.0

    for batch_id, (x_val, y_val_ans) in enumerate(zip(x_val_data, y_val)):
        x_val, y_val_ans = x_val.to(device), y_val_ans.to(device)
        pre_val = model(x_val)

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

```

#### 模型预测

```python
@torch.no_grad()
def test(model_path, x_test, y_test, threshold):
    model = torch.load(model_path)
    model.eval()
    right = 0
    # 测试集测试准确率
    for pok, ans in zip(x_test, y_test):
        pre = model(pok)
        if np.abs((pre - ans).cpu()) <= threshold and 0 <= pre <= 1:
            right += 1
    return right / len(y_test)

```

#### 重要参数

```python
# 概率与实际值差的阈值(0~1)，阈值越小，训练越严
threshold = 0.4

# 训练轮数
epochs = 50

# 学习率（梯度下降的步长）
learning_rate = 1e-3

# 极小值（避免 Adam 优化器计算时分母趋近于0）
eps = 1e-3
# L2-regularization（Adam优化器内，使权重参数尽可能小，避免参数互相依赖，也使梯度下降更加平滑）
weight_decay = 1e-3

```


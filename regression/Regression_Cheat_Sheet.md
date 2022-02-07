# Regression Cheat Sheet

> @ wyfffffei
>
> 示例数据集：<https://www.kaggle.com/harlfoxem/housesalesprediction>



## Data Init

### 数据检查

```python
import pandas as pd
import numpy as np
data = pd.read_csv("xxx.csv")
pd.options.display.max_columns = 25

# -> (数据总数, 每笔数据包含的信息数)
print(data.shape)

# 默认输出前五条
print(data.head())

# 检查数据类型
print(data.dtypes)
```

### 数据预处理

```python
# 数据拆分: date -> year + month
data["year"] = pd.to_numeric(data["date"].str.slice(0, 4))
data["month"] = pd.to_numeric(data["date"]).str.slice(4, 6)

# 删除多余数据 ('inplace'指删除源数据)
data.drop(["id"], axis="columns", inplace=True)
data.drop(["date"], axis="columns", inplace=True)

# 数据分割
# 获取数据索引并打乱
data_num = data.shape[0]
indexs = np.random.permutation(data_num)

# 划分三类数据集的索引 (0.6 + 0.2 + 0.2)
train_indexs = indexs[:int(data_num * 0.6)]
val_indexs = indexs[int(data_num * 0.6) : int(data_num * 0.8)]
test_indexs = indexs[int(data_num * 0.8):]

# 通过索引取出数据
train_data = data.loc[train_indexs]
val_data = data.loc[val_indexs]
test_data = data.loc[test_indexs]
```



### 归一化（Normalization）

 ![feature-scaling.png](./img/feature-scaling.png)

```python
# Standard Score (z-score)
# mean: 平均值
# std: 标准差
# x_norm = (x - mean) / std

train_validation_data = pd.concat([train_data, val_data])
mean = train_validation_data.mean()
std = train_validation_data.std()

train_data = (train_data - mean) / std
val_data = (val_data -mean) / std
```



### 建立 Numpy 格式数据集

```python
# 训练输入值和预计输出值
x_train = np.array(train_data.drop("price", axis="columns"))
y_train = np.array(train_data["price"])

# 验证数据集同上
# x_val = ...
# ...

# 查看训练数据集总数
print(x_train.shape)
```





## Model Create

```python
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
    
# func = FullConnection()
# input = torch.randn(784, requires_grad=True)
# out = func(input)
# print(out)

device = torch.device("cpu")
net = FullConnection().to(device)
learning_rate = 1e-2
epochs = 100
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss().to(device)
train_data, test_data = dataloader()
logger = SummaryWriter("./logs_train")
```





## Train the Model (step)

```python
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
```





## Test the Model (step)

```python
net.eval()
test_loss = 0
accurancy = 0
with torch.no_grad():
    for tes_data, tes_target in test_data:
        tes_data, tes_target = tes_data.to(device), tes_target.to(device)
        pre_test = net(tes_data)
        test_loss += loss_func(pre_test, tes_target).item()
        accurancy += (pre_test.argmax(1) == tes_target).sum()
test_loss /= len(test_data
print("\nTest set : Average loss: {:.4f}, Accurancy: {}/{}({:.3f}%)".format(
    test_loss, accurancy, len(test_data), 100.*accurancy/len(test_data)
))
logger.add_scalar("test_loss", test_loss, epoch)
logger.add_scalar("test_accuracy", accurancy, epoch)
```





## Other Questions

### 过拟合(overfitting)

*Q:训练的网络模型对验证数据集性能很差，但对训练数据集性能很好*

*A:模型太复杂或者训练数据太少*

 ![overfitting.png](./img/overfitting.png)

#### 缩减模型大小

#### 加入权重正则化(Weights Regularization)

 ![weight_regularization.png](./img/weight_regularization.png)

 ![weight_regularization_2.png](./img/weight_regularization_2.png)

#### 加入Dropout





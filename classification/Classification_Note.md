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



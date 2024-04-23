---
title: "理解和构建用于MNIST分类的卷积神经网络"
date: 2024-04-23T10:56:17+08:00
categories: ["AI"]
tags: [AI]
---
在深度学习领域，构建神经网络来解决各种任务是一项令人兴奋的工作。在本文中，我们将深入探讨使用PyTorch构建卷积神经网络（CNN）对来自流行的MNIST数据集的手写数字进行分类。

## 1、导入库和加载数据

首先，让我们通过导入必要的库和加载MNIST数据集来设置我们的环境。PyTorch和torchvision对于处理数据和创建神经网络至关重要，而matplotlib则有助于可视化图像。

```python
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

现在，让我们加载数据集。我们将对数据进行归一化处理，以使其均值为零，方差为1，以确保训练稳定性。

```python
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=64, shuffle=True)
```

## 2、数据可视化

在深入网络架构之前，让我们先偷偷看一下我们的数据。可视化一些样本图像可以帮助我们了解数据集的特征。

```python
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
```

## 3、定义神经网络架构

现在是核心部分 - 定义我们的CNN架构。我们将为数字分类创建一个简单而有效的网络。

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
```

## 4、前向传播和损失计算

在定义网络后，让我们看看它如何处理一批输入图像并计算损失。

```python
image = images[:2]
label = labels[:2]
out = net(image)
criterion = nn.CrossEntropyLoss()
loss = criterion(out, label)
print(loss)
```

## 5、训练模型

现在是训练时间！我们将遍历数据集多个周期，更新模型参数以最小化损失。

```python
optimizer = optim.SGD(net.parameters(), lr=0.01)

def train(epoch):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

train(1)
```

## 6、评估模型性能

最后，让我们评估我们训练好的模型在测试集上的表现如何。

```python
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('网络在10000个测试图像上的准确率为：%d %%' % (100 * correct / total))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/ed9e97a5617d4acb881ea30a43c35337.png)

通过跟随这些步骤，我们成功地构建并训练了一个用于MNIST数字分类的CNN，在未见过的数据上取得了不错的准确率。这为更高级的深度学习工作奠定了基础。祝编码愉快！ 🚀

以下是完整的代码：

```python
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 导入数据集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=64, shuffle=True)


def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(train_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))


# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)


# 前向传播
image = images[:2]
label = labels[:2]
out = net(image)
print(out)


# 计算损失
criterion = nn.CrossEntropyLoss()
loss = criterion(out, label)
print(loss)


# 反向传播与更新参数
optimizer = optim.SGD(net.parameters(), lr=0.01)
image = images[:2]
label = labels[:2]
optimizer.zero_grad()
out = net(image)
loss = criterion(out, label)
loss.backward()
optimizer.step()


# 开始训练
def train(epoch):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0


train(1)


# 观察模型预测效果
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('网络在10000个测试图像上的准确率为：%d %%' % (100 * correct / total))
```

这段代码涵盖了从数据加载、构建模型、训练到评估的完整流程，使用了PyTorch库进行实现。

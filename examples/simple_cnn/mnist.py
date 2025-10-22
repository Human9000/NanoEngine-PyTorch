import os
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src import *


class DoubleConvBlock(nn.Module):
    def __init__(self):
        super(DoubleConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        # 第二个卷积层: 输入通道=10, 输出通道=20, 卷积核大小=5
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=5)
        # Dropout层, 防止过拟合
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层: 输入通道=1 (灰度图), 输出通道=10, 卷积核大小=5
        self.conv = DoubleConvBlock()
        # 第一个全连接层
        self.fc1 = nn.Linear(320, 100)  # 320 是根据卷积和池化后的尺寸计算得出的
        self.fc2 = nn.Linear(100, 100)  # 320 是根据卷积和池化后的尺寸计算得出的
        # 第二个全连接层 (输出层)
        self.fc3 = nn.Linear(100, 10)  # 10个类别 (0-9)

    # 定义前向传播的过程
    def forward(self, x):
        x = self.conv(x)
        # "压平"张量，为全连接层做准备
        x = x.view(-1, 320)
        # 全连接层 -> ReLU激活
        x = F.relu(self.fc1(x))
        # Dropout
        x = F.dropout(x, training=self.training)
        # 输出层
        x = self.fc2(x)
        x = self.fc3(x)
        # 使用 log_softmax 作为输出，配合 NLLLoss 损失函数
        return F.log_softmax(x, dim=1)


def train(epoch):
    model.train()  # 将模型设置为训练模式
    loss_sum = 0
    loss_n = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移动到指定设备
        data, target = data.to(device), target.to(device)

        # 1. 清空梯度
        optimizer.zero_grad()

        # 2. 前向传播
        output = model(data)

        # 3. 计算损失
        loss = criterion(output, target)

        loss_sum += loss.item()
        loss_n += 1
        # 4. 反向传播
        loss.backward()

        # 5. 更新权重
        optimizer.step()

        if batch_idx % 100 == 0:
            print()
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            print()
    return loss_sum/ loss_n

def test():
    model.eval()  # 将模型设置为评估模式
    test_loss = 0
    correct = 0
    # 在评估阶段,我们不需要计算梯度
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 累加批次损失
            test_loss += criterion(output, target).item()
            # 获取预测结果中概率最大的那个类别的索引
            pred = output.argmax(dim=1, keepdim=True)
            # 累加预测正确的数量
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')

    # 定义数据预处理
    # transforms.ToTensor() 会将 PIL Image 或 numpy.ndarray 转换为 FloatTensor，并将像素值从 [0, 255] 缩放到 [0.0, 1.0]
    # transforms.Normalize() 会对张量进行标准化，使其均值为0，标准差为1


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 下载训练数据集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 下载测试数据集
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 创建数据加载器 (DataLoader)
    # DataLoader 可以帮助我们轻松地实现数据的批量加载、打乱等操作
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    print("数据加载完成!")

    # 实例化模型并移到GPU (如果可用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    print("模型定义完成:\n", model)
    # 转成图结构
    model = translate_model_to_GraphModule(model)
    print(model.state_dict().keys())

    model = model.to(device)

    # 定义损失函数
    criterion = nn.NLLLoss()

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-2,   weight_decay=1e-4)

    print("损失函数和优化器定义完成!")

    if not os.path.exists("./model.pt"):
        # 运行训练
        epochs = 5
        for epoch in range(1, epochs + 1):
            train(epoch)

        print("\n训练完成!")
        torch.save(model.state_dict(), "./model.pt")

    # 运行测试
    test()

    state_dict = torch.load("./model.pt")
    model.load_state_dict(state_dict, strict=False)
    # 添加量化
    model = graph_module_insert_quantization_nodes(
        model,
        customer_rules=[
            QRule(r".weight$", 1, None, None),
            # QRule(r"fc2\.weight$", 1, None, None),
        ],
    ).to(device)
    print(model)
    model.graph.print_tabular()
    # 运行训练
    epochs = 5
    for epoch in range(1, epochs + 1):
        if train(epoch) < 1e-6:
            break
    print("\n训练完成!")
    # 运行测试
    test()

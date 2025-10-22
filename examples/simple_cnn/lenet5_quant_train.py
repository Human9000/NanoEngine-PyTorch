import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import transforms

import torch.nn.functional as F
import time
import sys

sys.path.append("./NanoEngine-Pytorch-v0.02")
from nept import (
    translate_model_to_GraphModule,
    QRule,
    graph_module_insert_quantization_nodes,
    remove_quantization_nodes,
    ex_quantize_model_fully_encapsulated,
    sanitize_onnx_names,
    shape_inference,
)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.bn = nn.BatchNorm2d(16)

    # 定义前向传播路径
    def forward(self, x):
        # 激活函数使用现代常用的 ReLU 替代 Sigmoid/Tanh

        # C1 -> S2
        # x = self.pool1(F.relu(self.conv1(x)))
        # # C3 -> S4
        # x = self.pool2(F.relu(self.conv2(x)))

        # C1 -> S2
        x = F.relu(self.conv1(x))
        # C3 -> S4
        x = F.relu(self.conv2(x))
        x = self.bn(x)

        # 展平特征图 (Flatten)
        x = x.view(-1, 16 * 5 * 5)

        # F5 (C5)
        x = F.relu(self.fc1(x))
        # F6
        x = F.relu(self.fc2(x))
        # Output
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, data_loader, optimizer, criterion, epochs=1):
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Stip [{i + 1}/{len(data_loader)}], Loss:{loss.item():.4f}, {(time.time() - start_time):.2f}s")
                start_time = time.time()
                print("\n--- 保存量化感知模型 ---")
                torch.save(model.state_dict(), "lenet5_quant.pth")


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy


if __name__ == "__main__":
    # # model = QuantizableLeNet5()
    model = LeNet5()
    quantized_gm = translate_model_to_GraphModule(model)
    print("--- 图结构模型 ---")
    print(quantized_gm)
    print(list(quantized_gm.state_dict().keys()))


    # 2. Trace the model and insert quantization nodes
    quantized_gm = graph_module_insert_quantization_nodes(
        quantized_gm,
        customer_rules=[
            # QRule(r"conv1\.weight", 4, 0.1, 0, True),
            # QRule(r"conv2\.weight", 4, 0.1, None),
            # QRule(r"fc1\.weight", 1, 0.1, 0, False),
            # QRule(r"fc1\.weight", 1, 0.1, 0, False),
            # QRule(r"fc2\.weight", 1, 0.1, 0, False),

            QRule(r".*?", 8, 0.1, 1, True),
            # QRule(r"^x$", 8, 0.1, 1, True),
            # QRule(r"conv", 8, 0.1, 1, True),
            # QRule(r"linear", 8, 0.1, 1, True),
            # QRule(r"avg_pool", 8, 0.1, 1, True),
            # QRule(r"batch_norm", 8, 0.1, 1, True),

            QRule(r"\.weight$", 1, 0.1, 0, True),
            QRule(r"\.bias$", 1, 0.1, 0, True),
            QRule(r"\.running_var$", 1, 0.1, 0, False),
            QRule(r"\.running_mean", 1, 0.1, 0, False),
            # QRule(r"bn\.weight", 8, 0.1, 0, False),
            # QRule(r"fc3\.weight", 8, 0.1, None),
            # {"pattern": r"0\.conv\.conv1\.weight", "bits_len": 4, "lr": 0.01, "channel_dim": 0},  # 自定义规则
        ],
    )
    # QLeNet5
    print("\n--- 量化后的模型结构 (完全封装) ---")
    # print(quantized_gm)
    # model = QLeNet5()
    # quantized_gm = translate_model_to_GraphModule(model)
    print(quantized_gm)
    quantized_gm.graph.print_tabular()
    # print(model)
    # 加载 MNIST 数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 实例化新的量化感知模型
    # model = model.to(device)
    model = quantized_gm.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("--- 开始量化感知训练 (QAT) ---")
    train(model, train_loader, optimizer, criterion, epochs=10)

    print("\n--- 评估 QAT 模型 ---")
    evaluate(model, test_loader)

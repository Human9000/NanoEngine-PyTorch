import torch
import torch.nn as nn
import torch.optim as optim

import tqdm
import torchvision
from torchvision import transforms
from torchvision.datasets import VOCDetection
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.nn.functional as F
import time
import sys

sys.path.append("./NanoEngine_Pytorch")
from nept import (
    translate_model_to_GraphModule,
    QRule,
    graph_module_insert_quantization_nodes,
    remove_quantization_nodes,
    ex_quantize_model_fully_encapsulated,
    sanitize_onnx_names,
    shape_inference,
)

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.datasets import VOCDetection
from collections import OrderedDict
import os
import time
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet18, ResNet18_Weights
from collections import OrderedDict
import xml.etree.ElementTree as ET  # 用于解析 XML 标注文件
import timm

# VOC 20 个类别 (索引 0-19)
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(VOC_CLASSES)}
NUM_CLASSES = len(VOC_CLASSES)  # 20


def voc_xml_to_multilabel(target_xml_dict):
    """
    将 VOCDetection 返回的 XML 字典转换为多标签 Tensor (Multi-Hot Vector)。
    """
    # 1. 初始化一个长度为 20 的零向量 (Multi-Hot Vector)
    labels = torch.zeros(NUM_CLASSES, dtype=torch.float32)

    # 2. 从字典中提取根元素
    # VOCDetection 返回的 target 已经是 XML 解析后的字典
    root = target_xml_dict['annotation']

    # 3. 遍历所有物体
    if 'object' in root:
        objects = root['object']
        # 确保 objects 是一个列表 (单个物体时可能不是列表)
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            class_name = obj['name']
            if class_name in CLASS_TO_IDX:
                idx = CLASS_TO_IDX[class_name]
                labels[idx] = 1.0  # 标记该类别存在

    return labels


# 标准的分类任务 Transforms
transform_classification = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一尺寸
    # transforms.Resize((28, 28)), # 统一尺寸
    transforms.ToTensor(),  # 转换为 Tensor (HWC -> CHW, 0-255 -> 0-1)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet 均值
        std=[0.229, 0.224, 0.225]  # ImageNet 标准差
    )
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def train(model, data_loader, optimizer, criterion, epochs=1):
#     model.train()
#     start_time = time.time()
#     for epoch in range(epochs):
#         for i, (images, labels) in enumerate(data_loader):
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             # print(outputs.shape, labels.shape)
#
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             if (i + 1) % 100 == 0:
#                 print(f"Epoch [{epoch + 1}/{epochs}], Stip [{i + 1}/{len(data_loader)}], Loss:{loss.item():.4f}, {(time.time() - start_time):.2f}s")
#                 start_time = time.time()


def multi_loss(pred: torch.Tensor, gt: torch.Tensor, p=0.5):
    gt0 = 1 - gt
    gt1 = gt
    err = (gt - pred).pow(2)
    err_p = ((err * gt1).sum(dim=0) / (gt1.sum(dim=0) + 1)).mean()  # 正样本，每个类的，正样本均衡
    err_n = ((err * gt0).sum(dim=0) / (gt0.sum(dim=0) + 1)).mean()  # 负样本，每个类的，负样本均衡
    # err_p = (((gt - pred).pow(2) * gt).sum(dim=0) / (gt.sum(dim=0) + 1)).mean()  # 正样本，每个类的，正样本均衡
    # err_n = (((gt - pred).pow(2) * (1 - gt)).sum(dim=0) / ((1 - gt).sum(dim=0) + 1)).mean()  # 负样本，每个类的，负样本均衡
    loss = err_p * p + err_n * (1 - p)  # mean代表类间均衡，+代表类内均衡
    return loss * 2


def train_one_epoch(model, data_loader, optimizer, lr_sc):
    model.train()
    start_time = time.time()
    sum_loss: float = 0.
    loss_n = 0
    tqdm_train_loader = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Training")
    for i, (images, labels) in tqdm_train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        # print(outputs.shape, labels.shape)
        # 正样本 = (labels >= 0.5)
        # 负样本 = ~正样本
        #
        # loss_pos = (outputs[正样本] - 0.8).pow(2).mean()
        # loss_neg = (outputs[负样本] - 0.1).pow(2).mean()
        # loss = loss_pos + loss_neg

        loss = multi_loss(outputs, labels)

        # loss = criterion(outputs, labels)

        sum_loss += loss.item()
        loss_n += 1
        loss.backward()
        optimizer.step()
        lr_sc.step()
        tqdm_train_loader.set_description("Training Loss: {:.4f}".format(sum_loss / loss_n))
        # if i > 40:
        #     break
        # break

    return sum_loss / loss_n, time.time() - start_time
    # if (i + 1) % 100 == 0:
    #     print(f"Epoch [{epoch + 1}/{epochs}], Stip [{i + 1}/{len(data_loader)}], Loss:{loss.item():.4f}, {(time.time() - start_time):.2f}s")
    #     start_time = time.time()


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        i = 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 1. 对模型 Logits 应用 Sigmoid 激活函数
            probabilities = torch.sigmoid(outputs)

            # 2. 应用阈值 (0.5) 转换为二元预测 (0 或 1)
            # 这一步将 [B, 20] 的连续值转换为 [B, 20] 的二元 Tensor

            predicted = probabilities

            # 3. 计算每个样本的正确预测（全部 20 个标签都正确才算一个）
            # 对于多标签任务，通常使用 Exact Match Ratio (所有标签都预测正确) 或 F1 Score
            # 如果您想计算 Exact Match Ratio：
            total_correct_predictions = ((probabilities > 0.5).int() == labels).sum().item()  # 计算所有标签的匹配数

            # 您的原始代码可能想计算所有标签位置的准确率：
            correct += total_correct_predictions
            total += labels.numel()  # Batch Size * 20
            i += 1
            if i > 100:
                break

    accuracy = 100 * correct / total
    return accuracy, images, labels, predicted


if __name__ == '__main__':
    ROOT_DIR = 'Z:/data'  # 你的 VOC 数据集存放目录，例如 VOCdevkit/VOC2012

    # 1. 训练集
    train_dataset = datasets.VOCDetection(
        root=ROOT_DIR,
        year='2007',  # 也可以是 '2007' 或 '2007', '2012'
        image_set='trainval',  # 使用 trainval 进行训练
        download=False,  # 如果已下载，设为 False
        transform=transform_classification,  # 应用图像转换
        target_transform=voc_xml_to_multilabel  # 💥 应用自定义 Target 转换
    )

    # 2. 验证集
    val_dataset = datasets.VOCDetection(
        root=ROOT_DIR,
        year='2007',
        image_set='test',
        download=False,
        transform=transform_classification,
        target_transform=voc_xml_to_multilabel
    )

    # 3. DataLoader (分类任务可使用默认 collate_fn)
    BATCH_SIZE = 32
    NUM_WORKERS = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        # 分类任务的图像尺寸相同，可使用默认 collate_fn
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    print(f"训练集大小: {len(train_dataset)}")
    # --- 2. 模型、优化器设置 ---
    model = timm.create_model('resnet50', pretrained=False, num_classes=20)
    # quantized_gm = translate_model_to_GraphModule(model)
    print("--- 图结构模型 ---")
    # print(quantized_gm)
    # print(list(quantized_gm.state_dict().keys()))

    # 2. Trace the model and insert quantization nodes
    # quantized_gm = graph_module_insert_quantization_nodes(
    #     quantized_gm,
    #     customer_rules=[
    #         QRule(r"weight$", 8, 0.01, 0, True),  # 自定义规则
    #         QRule(r"running_var$", 8, 0.01, 0, True),  # 自定义规则
    #     ],
    # )

    print("\n--- 量化后的模型结构 (完全封装) ---")
    # print(quantized_gm)
    # quantized_gm.graph.print_tabular()
    model = model.to(device)
    # 优化器

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-8)

    lr_sc = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=110, eta_min=1e-6)

    # 加载权重
    model.load_state_dict(torch.load("resnet18_voc.pth"))

    print("--- 开始量化感知训练 (QAT) ---")
    epochs = 2
    for epoch in range(epochs):
        avg_loss, time_use = train_one_epoch(model, train_loader, optimizer, lr_sc)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss:{avg_loss:.4f}, {time_use:.2f}s")
        # 保存权重
        torch.save(model.state_dict(), f"resnet18_voc.pth")
        print("Save Model: resnet18_voc.pth")

    print("\n--- 评估 QAT 模型 ---")
    accuracy, images, labels, predicted = evaluate(model, val_loader)
    print(f"Accuracy: {accuracy:f}%")

    import matplotlib.pyplot as plt
    import numpy as np

    plt.imshow(images[0].permute(1, 2, 0).cpu())
    l = np.where(labels[0].cpu().numpy() > 0.5)
    p = np.where(predicted[0].cpu().numpy() > 0.5)
    plt.title(f"GT: {l}, Pred: {p}")
    print(labels[0])
    print(predicted[0])
    print(1.0 * (predicted[0] > 0.5) == labels[0])
    plt.show()

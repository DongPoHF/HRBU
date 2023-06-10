# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage import io
import skimage.transform as tr
import warnings
import copy

plt.style.use('ggplot')
warnings.filterwarnings('ignore')
train_dir = 'datasets/datasets/train/'
test_dir = 'datasets/datasets/test/'

# 图像处理
train_transform = transforms.Compose([

    transforms.Resize((96, 96)),  # 调整大小
    # transforms.RandomCrop((224,224)),# 随机剪裁
    transforms.RandomHorizontalFlip(p=0.5),  # 随机翻转
    transforms.ColorJitter(
        brightness=(0.7, 1.3),  # 明暗程度
        contrast=(0.7, 1.3),  # 对比度
        saturation=(0.7, 1.3),  # 饱和度
        hue=(-0.05, 0.005),  # 颜色

    ),
    # transforms.RandomRotation(30, center=(0, 0), expand=True),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='symmetric'),
    # transforms.Grayscale(num_output_channels=3),
    # transforms.RandomAffine(degrees=0, shear=90, fillcolor=(255, 0, 0))
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
test_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
# 创建ds
train_ds = torchvision.datasets.ImageFolder(
    train_dir,
    transform=train_transform,
)
test_ds = torchvision.datasets.ImageFolder(
    test_dir,
    transform=test_transform,
)

# 创建dl
BATCHSIZE = 2
train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=BATCHSIZE,
    shuffle=True
)
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCHSIZE
)

# class TestNetBN(nn.Module):
#     def __init__(self):
#         super(TestNetBN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 3)
#         # 初始化一个BN层 deep :16
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, 3)
#         # 初始化第二个BN层 deep :32
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 64, 3)
#         # 初始化第3个BN层 deep :64
#         self.bn3 = nn.BatchNorm2d(64)
#         self.fc1 = nn.Linear(64 * 14 * 14, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, 15)
#
#     def forward(self, x):
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # x = x.to(device)
#         # print(x.is_cuda)
#         x = F.relu(self.conv1(x))
#         x = self.bn1(x)
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv2(x))
#         x = self.bn2(x)
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv3(x))
#         x = self.bn3(x)
#         x = F.max_pool2d(x, 2)
#         x = x.view(-1, 64 * 14 * 14)
#         x = F.dropout(x)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = F.dropout(x)
#         x = self.fc3(x)
#         return x
#         # return x.size()


# imgs,labels = next(iter(test_dl))
# models = TestNetBN()
# models(imgs)
# 加载预训练模型
model = torchvision.models.vgg16(pretrained=True)

# 模型放入加速运算显卡GPU
# 冻结 卷积基
for param in model.features.parameters():
    param.requires_grad = False
# 设置分类器
model.classifier[-1].out_features = 9
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#exp_lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
# 学习率衰减
# 参数1，优化器
# 参数2，间隔epoch数
# 参数3,衰减次数
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


# 搭建训练步骤
def train(dataloader, model, loss_fn, optimizer):
    # acc 当前样本一共有多少个
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # 初始化loss值
    train_loss, correct = 0, 0
    model.train()  # 模式为训练模式
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        # 前向传播（推理）
        pred = model(X)
        # 产生误差
        loss = loss_fn(pred, y)  # 返回平均误差
        # loss = torch.nn.functional.cross_entropy(pred, y)
        # 归零权重(固定写法)
        optimizer.zero_grad()
        # 反向传递
        loss.backward()
        # 利用梯度下降优化参数
        optimizer.step()
        # # 放弃张量微分的追踪（小验证）
        with torch.no_grad():
            train_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    exp_lr_scheduler.step()
    lr_res = optimizer.state_dict()['param_groups'][0]['lr']
    correct /= size
    train_loss /= num_batches
    return train_loss, correct


# 搭建测试 （test）
def test(dataloader, model):
    # acc 当前样本一共有多少个
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()  # 推理模式
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    test_loss /= num_batches
    return test_loss, correct


epochs = 50
train_acc = []
train_loss = []
test_acc = []
test_loss = []
for epoch in range(epochs):
    start = time.time()
    epoch_loss, epoch_acc = train(dataloader=train_dl, model=model, loss_fn=loss_fn, optimizer=optimizer)
    epoch_test_loss, epoch_test_acc = test(dataloader=test_dl, model=model)
    end = time.time()
    times = end - start
    train_acc.append(epoch_acc)
    train_loss.append(epoch_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)
    print('训练epoch{},训练集损失值:{:.2f},训练集的准确率:{:.2f}%,测试集损失值:{:.2f},测试集的准确率:{:.2f}%,消耗时间：{:.2f}s'.
          format(epoch + 1, epoch_loss, epoch_acc * 100, epoch_test_loss, epoch_test_acc * 100, times))
print('Train Done!!!!')
torch.save(model.state_dict(), 'vgg16.pt')
print('ok')

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import warnings

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


class Vgg16_net(nn.Module):
    def __init__(self):
        super(Vgg16_net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            # 意思是是否将计算得到的值覆盖之前的值，比如
            nn.ReLU(inplace=True),
            # 意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，
            # 这样能够节省运算内存，不用多存储其他变量
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # (32-3+2)/1+1=32    32*32*64
            # Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，
            # 一方面使得数据分布一致，另一方面避免梯度消失。
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (32-2)/2+1=16         16*16*64
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # (16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # (16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # (16-2)/2+1=8     8*8*128
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # (8-2)/2+1=4      4*4*256
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # (4-2)/2+1=2     2*2*512
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # (2-2)/2+1=1      1*1*512
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(
            # y=xA^T+b  x是输入,A是权值,b是偏执,y是输出
            # nn.Liner(in_features,out_features,bias)
            # in_features:输入x的列数  输入数据:[batchsize,in_features]
            # out_freatures:线性变换后输出的y的列数,输出数据的大小是:[batchsize,out_features]
            # bias: bool  默认为True
            # 线性变换不改变输入矩阵x的行数,仅改变列数
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 9)
        )

    def forward(self, x):
        x = self.conv(x)
        # 这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成512列
        # 那不确定的地方就可以写成-1
        # 如果出现x.size(0)表示的是batchsize的值
        # x=x.view(x.size(0),-1)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


# imgs,labels = next(iter(test_dl))
# models = TestNetBN()
# models(imgs)
# model = Vgg16_net()
# 加载预训练模型
model = torchvision.models.vgg16(pretrained=True)
# model.add_module("plain",
#                  torch.nn.Linear(224 * 224 * 3, 4096),
#                  torch.nn.ReLU(),
#                  torch.nn.Dropout(p=0.5),
#                  torch.nn.Linear(4096, 4096),
#                  torch.nn.ReLU(),
#                  torch.nn.Dropout(p=0.5),
#                  torch.nn.Linear(4096, 224 * 224 * 3)
#                  )
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(25088, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096, 9)
)
# print(model.parameters())
# 模型放入加速运算显卡GPU
# 冻结 卷积基
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for param in model.features.parameters():
    param.requires_grad = False
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


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


epochs = 3
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


def save_data():
    # train_loss, test_loss, train_acc, test_acc = res[0], res[1], res[2], res[3]
    np.savetxt('data/train_acc', train_acc)
    np.savetxt('data/train_loss', train_loss)
    np.savetxt('data/test_acc', test_acc)
    np.savetxt('data/test_loss', test_loss)


def plot_loss():
    ## 训练曲线可视化
    # 损失值 （越低越好）
    epoch = 50
    # train_loss, test_loss, train_acc, test_acc = res[0], res[1], res[2], res[3]
    train_loss_min_index = np.argmin(np.array(train_loss))
    train_loss_min_index_value = round(train_loss[train_loss_min_index], 6)
    test_loss_min_index = np.argmin(np.array(test_loss))
    test_loss_min_index_value = round(test_loss[test_loss_min_index].item(), 6)
    s = train_loss_min_index_value
    s_test = test_loss_min_index_value
    plt.plot(range(1, epoch + 1), train_loss, label='train_loss')
    plt.plot(range(1, epoch + 1), test_loss, label='test_loss')
    plt.text(train_loss_min_index + 1, train_loss_min_index_value, s)
    plt.text(test_loss_min_index + 1, test_loss_min_index_value, s_test)
    plt.plot(train_loss_min_index, train_loss_min_index_value, 'ro')
    plt.plot(test_loss_min_index, test_loss_min_index_value, 'bo')
    plt.title('vgg Loss values')
    plt.legend()
    plt.savefig('pic/train_acc.jpg')


def plot_acc():
    ## 训练曲线可视化
    # 准确率
    epoch = 50
    # train_loss, test_loss, train_acc, test_acc = res[0], res[1], res[2], res[3]
    train_acc_max_index = np.argmax(np.array(train_acc))
    train_acc_max_value = round(train_acc[train_acc_max_index], 6)
    test_acc_max_index = np.argmax(np.array(test_acc))
    test_acc_max_value = round(test_acc[test_acc_max_index], 6)
    s = train_acc_max_value
    s_test = test_acc_max_value
    plt.plot(range(1, epoch + 1), train_acc, label='train_acc')
    plt.plot(range(1, epoch + 1), test_acc, label='test_acc')
    plt.text(train_acc_max_index, train_acc_max_value, s)
    plt.text(test_acc_max_index, test_acc_max_value, s_test)
    plt.plot(train_acc_max_index + 1, train_acc_max_value, 'bo')
    plt.plot(test_acc_max_index + 1, test_acc_max_value, 'go')
    plt.title('vgg Acc values')
    # plt.ylim(0.5,1.08)
    # plt.xlim(0,31)
    plt.legend()
    plt.savefig('pic/train_acc.jpg')


save_data()
plot_loss()
plot_acc()

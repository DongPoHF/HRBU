from datetime import time

import numpy as np
import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

BATCH_SIZE = 2
LEARNING_RATE = 0.001
EPOCH = 3
N_CLASSES = 9

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

trainData = dsets.ImageFolder('datasets/datasets/train/', transform)
testData = dsets.ImageFolder('datasets/datasets/test/', transform)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


class VGG16(tnn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7 * 7 * 512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = tnn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out


vgg16 = VGG16(n_classes=N_CLASSES)
vgg16.cuda()

# Loss, Optimizer & Scheduler
cost = tnn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

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
# # Train the model
# for epoch in range(EPOCH):
#
#     avg_loss = 0
#     cnt = 0
#     for images, labels in trainLoader:
#         images = images.cuda()
#         labels = labels.cuda()
#
#         # Forward + Backward + Optimize
#         optimizer.zero_grad()
#         _, outputs = vgg16(images)
#         loss = cost(outputs, labels)
#         avg_loss += loss.data
#         cnt += 1
#         print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss / cnt))
#         loss.backward()
#         optimizer.step()
#     scheduler.step(avg_loss)
#
# # Test the model
# vgg16.eval()
# correct = 0
# total = 0
#
# for images, labels in testLoader:
#     images = images.cuda()
#     _, outputs = vgg16(images)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted.cpu() == labels).sum()
#     print(predicted, labels, correct, total)
#     print("avg: %f" % (100 * correct / total))
#
# # Save the Trained Model
# torch.save(vgg16.state_dict(), 'cnn.pkl')

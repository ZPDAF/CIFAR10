import torch
import torchvision
import argparse
from torch import nn
from torch.utils.data import DataLoader

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='../data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)


print("训练集的长度:{}".format(len(train_data)))
print("测试集的长度:{}".format(len(test_data)))

# DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x
def train_test(args):
    # 创建网络模型
    model = Model().cuda()

    # 损失函数
    loss = nn.CrossEntropyLoss().cuda()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    i = 1

    # 记录最大准确率
    max_accuracy = 0

    # 开始训练
    for epoch in range(args.epochs):
        num_time = 0
        print('开始第{}轮训练'.format(epoch + 1))
        model.train()
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            output = model(imgs)
            loss_in = loss(output, targets)
            optimizer.zero_grad()
            loss_in.backward()
            optimizer.step()
            num_time += 1

        sum_loss = 0  # 记录总体损失值

        accurate = 0
        model.eval()
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.cuda()
                targets = targets.cuda()
                output = model(imgs)
                loss_in = loss(output, targets)

                sum_loss += loss_in
                accurate += (output.argmax(1) == targets).sum()

        print('第{}轮测试集的损失:{} 正确率:{:.2f}%'.format(epoch + 1, loss_in, accurate / len(test_data) * 100))

        if accurate > max_accuracy:
            max_accuracy = accurate

        i = i + 1
    print('最大准确率为{:.2f}%'.format(max_accuracy / len(test_data) * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    args = parser.parse_args()
    train_test(args)
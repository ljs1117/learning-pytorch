# https://www.bilibili.com/video/BV1Y7411d7Ys?p=11 第11讲例程2
import torch
from torchvision import transforms  # 用于数据处理
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])
# 这个变化包含两步，第一步将PIL Image转换为Pytorch Image
# 第二步归一化，采用的均值和方差是MNIST整个数据集的均值和方差

train_dataset = datasets.MNIST(
    root="dataset/mnist/", train=True, download=True, transform=transform)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
# num_worker默认为0

test_dataset = datasets.MNIST(
    root="dataset/mnist/", train=False, download=True, transform=transform)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ResidualBlock为了解决梯度消失的问题
# 要求输入和输出的维度相同，且需要先做加法再激活


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv = torch.nn.Conv2d(
            channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv(x))
        y = self.conv(y)
        return F.relu(x+y)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)

        self.rb1 = ResidualBlock(16)
        self.rb2 = ResidualBlock(32)

        self.maxpool = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        # x为(N=64,1,28,28)
        batch_size = x.size(0)
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.rb1(x)
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.rb2(x)
        x = x.view(batch_size, -1)
        # -1可自动计算为320
        x = self.fc(x)

        return x
        # 最后一层不激活，SoftMax含在CrossEntropyLoss中


model = Net()
criterion = torch.nn.CrossEntropyLoss()
# 默认reduction="mean"
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# 带动量的SGD，可解决鞍点和峡谷（粗糙的梯度估计使其撞向两测峭壁）问题，也可加速
# v_t=momentum*v_(t-1)-lr*g_t
# 更新步长不仅取决于梯度和学习率，还取决于历史梯度的大小和方向


def train(epoch):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每300个batch输出一次loss，loss是这300的平均
        running_loss += loss.item()
        if i % 300 == 299:
            print('[%d,%5d] loss:%.3f' % (epoch+1, i+1, running_loss/300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            outputs = model(inputs)
            prediction = torch.max(outputs.data, dim=1)[1]
            # dim:0表示列，1表示行
            # 输出每行最大值以及索引 [0]为最大值 [1]为索引
            total += labels.size(0)
            correct += torch.eq(prediction, labels).sum().item()

    print('Accuracy on test_set:%d %%' % (100*correct/total))


if __name__ == "__main__":
    for epoch in range(10):
        train(epoch)
        test()

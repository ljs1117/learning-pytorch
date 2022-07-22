
# https://www.bilibili.com/video/BV1Y7411d7Ys?p=8 第八讲例程
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 注意：该例程收敛效果并不好，考虑以下做法
# 1、增大batch_size,易收敛，且速度提高，但可能陷入局部最优解
# 2、改变学习率
# 3、使网络结构更复杂

# 本例构建数据集采用init将所有数据导入内存，getitem读出


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(
    dataset=dataset, batch_size=32, shuffle=True, num_workers=0)
# num_workers提高读取速度,原理：
# num_workers!=0时，dataloader有多个worker将其负责的batch加载进主存
# num_workers=0,dataloader,dataloader不再有自主加载数据到RAM这一步骤，而是在主存中直接找batch，找不到时再加载相应的batch，只有主进程自己去加载数据


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.ReLU = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        # torch.sigmoid()为函数
        # torch.nn.Sigmoid()为类，一般用于层

    def forward(self, x):
        x = self.ReLU(self.linear1(x))
        x = self.ReLU(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        # 本例试用ReLU函数，但最后一层用sigmoid保证概率[0,1]
        return x


model = Model()
criterion = torch.nn.BCELoss(reduction="mean")
# "sum"对每个样本loss求和，“mean”为对每个样本Loss求平均，“none”为保留Loss矩阵
# weight (Tensor, optional) – a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch.

# 本例试用Adam优化器
optimizer_Adam = torch.optim.Adam(model.parameters(), lr=0.01)

if __name__ == '__main__':
    loss_list = []
    for epoch in range(100):
        for i, (inputs, labels) in enumerate(train_loader, 0):  # train_loader完成shuffle和loader
            # enumerate(,0)索引从0开始
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer_Adam.zero_grad()
            loss.backward()
            optimizer_Adam.step()
        loss_list.append(loss.item())

'''
    plt.plot(range(1, 101), loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Adam")
    plt.show()
'''

# 补充一种将现有数据集划分成训练集和测试集的方法
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

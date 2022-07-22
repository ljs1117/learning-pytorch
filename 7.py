# https://www.bilibili.com/video/BV1Y7411d7Ys?p=7 第七讲例程
import numpy as np
import torch
import matplotlib.pyplot as plt

xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])
# 此处[-1]保证torch中为矩阵，而不是向量


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

loss_list = []
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    # 二分类系统的评价指标
    # TP(正确的正例) FP（错误的正例，错检） TN FN（错误的反例，漏检）
    # 准确率accuracy=(TP+TN)/(P+N)预测正确的所占的比例
    # 精确率precision=TP/P 允许漏检不能错检
    # 召回率recall=TP/(TP+FN) 允许错检不能漏检
    # F_score=w*precision*recall/(precision+recall)

    # 加一个指标acc(准确率)
    y_pred_label = torch.where(
        y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
    # torch.where(condition,a,b)按一定规律生成tensor，满足条件选择a,否则选择b
    acc = torch.eq(y_pred_label, y_data).sum().item()/y_data.size(0)
    # touch.eq(a,b)逐元素比较，相同则在该位置返回1，torch.eq(a,b).sum().item()预测正确的个数
    # acc=预测正确的个数/总样本数
    print(epoch, "loss=", loss.item(), "acc=", acc)
    loss_list.append(loss.item())

    optimizer_Adam.zero_grad()
    loss.backward()
    optimizer_Adam.step()


plt.plot(range(1, 1001), loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Adam")
plt.show()

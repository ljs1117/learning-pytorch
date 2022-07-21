# https://www.bilibili.com/video/BV1Y7411d7Ys?p=6 第六讲例程
import torch
import numpy as np
import matplotlib.pyplot as plt

# torch.tensor()和torch.Tensor()区别
# torch.Tensor()是类，默认张量类型torch.FloatTensor()
# torch.tensor()是函数，其根据原始数据类型生成相应的torch.LongTensor、torch.FloatTensor和torch.DoubleTensor
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.], [0.], [1.]])
# 或写作以下形式
#y = np.array([[0], [0], [1]], dtype=np.float32)
#y_data = torch.tensor(y)
# 或写作
#y_data = torch.Tensor([[0], [0], [1]])


class LogisticRegressionModel(torch.nn.Module):
    # 注意该类可生成可调用对象，即存在__call__()函数
    # __call__()中存在forward()
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(reduction="sum")
# "sum"对每个样本loss求和，“mean”为对每个样本Loss求平均，“none”为保留Loss矩阵
# weight (Tensor, optional) – a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch.
optimizer_SGD = torch.optim.SGD(model.parameters(), lr=0.01)
# 可使用不同的优化器，例如Adam
#optimizer_Adam = torch.optim.Adam(model.parameters(), lr=0.01)

mse_list = []
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    mse_list.append(loss.item())

    optimizer_SGD.zero_grad()
    loss.backward()
    optimizer_SGD.step()


print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data.item())

plt.plot(range(1, 101), mse_list)
plt.xlabel("epoch")
plt.ylabel("mse")
plt.title("SGD")
plt.show()

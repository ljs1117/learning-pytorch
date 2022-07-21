# https://www.bilibili.com/video/BV1Y7411d7Ys?p=5 第五讲例程
import torch
import matplotlib.pyplot as plt

# 矩阵，一行为一个样本，一列为一个特征
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    # 注意该类可生成可调用对象，即存在__call__()函数
    # __call__()中存在forward()
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        # (1,1)表示输入x,y的特征的维度均为1维

    # 重写forward
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

criterion = torch.nn.MSELoss(reduction="sum")
# "sum"对每个样本loss求和，“mean”为对每个样本Loss求平均，“none”为保留Loss矩阵
optimizer_SGD = torch.optim.SGD(model.parameters(), lr=0.01)
# 注意由于x_data含3个元素，此处SGD为批量随机梯度下降
# model.parameters()初始化所有需要计算梯度的参数
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

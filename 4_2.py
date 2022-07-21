# https://www.bilibili.com/video/BV1Y7411d7Ys?p=4 第四讲练习
import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.tensor([1.0])
w2 = torch.tensor([1.0])
b = torch.tensor([1.0])
w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True


def forward(x):
    return x*x*w1+x*w2+b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2


print("predict (before training)", 4, forward(4).item())

mse_list = []
for epoch in range(100):
    cost = 0
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        cost += l.item()  # 注意一定要用.item(),l为tensor
        l.backward()
        print('\tgrad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    mse_list.append(cost/len(x_data))
    print('progress:', epoch, l.item())

print("predict (after training)", 4, forward(4).item())
plt.plot(range(1, 101), mse_list)
plt.xlabel("epoch")
plt.ylabel("mse")
plt.show()

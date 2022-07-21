# https://www.bilibili.com/video/BV1Y7411d7Ys?p=4 第四讲例程
import torch
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])
w.requires_grad = True  # 需要计算梯度的话，需设置

# 注意Tensor运算时，会建立计算图


def forward(x):
    return x*w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2


print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # l is a Tensor, forward, compute the loss
        l.backward()  # backward,compute grad for Tensor whose requires_grad set to True

        # 注意.item()和.data的区别
        # .data返回的是一个tensor；而.item()返回的是一个数。
        # .item()只用于只有一个元素的tensor
        # a  = torch.ones([1,3])则a有两个元素，a.data[0, 1].item()取第二个元素的值

        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        # The grad computed by .backward() will be accumulated.After update, remember set the grad to zero
        w.grad.data.zero_()

    print('progress:', epoch, l.item())

print("predict (after training)", 4, forward(4).item())

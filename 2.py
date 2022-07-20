
# 对应https://www.bilibili.com/video/BV1Y7411d7Ys/?p=2 第二讲作业
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
W = np.arange(0.0, 4.1, 0.1)
B = np.arange(-2.0, 2.1, 0.1)

# indexing
# if xv, yv = np.meshgrid(x, y, indexing='ij')
# treat xv[i,j], yv[i,j],相当于纵向为i，横向为j
# if xv, yv = np.meshgrid(x, y, indexing='xy')
# treat xv[j,i], yv[j,i]，相当于纵向为y，横向为x
# 但以上在该例程中一致

w, b = np.meshgrid(W, B, indexing='ij')

# 相当于矩阵运算


def forward(x):
    return x*w+b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2


l_sum = 0
# 注意zip用法，返回以元组为元素的列表
# zip（*z）为逆过程
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val)
    loss_val = loss(x_val, y_val)
    l_sum += loss_val

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(w, b, l_sum/3, cmap=plt.get_cmap("coolwarm"))
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("loss")
ax.text(0, 2, 45, "Cost Value")
plt.show()

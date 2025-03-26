# -*- coding: utf-8 -*-
"""
辅助程序，使用此程序，在PCA之后探索卷积核的构造方式
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

# 定义k1和k2
k1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
k1 = k1/np.sqrt(6)
k2 = k1.T.reshape(9, )
k1 = k1.reshape(9, )
k3 = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]])
k3 = k3/np.sqrt(18)
k4 = k3.T.reshape(9, )
k3 = k3.reshape(9, )
k7 = np.array([1, 0, -1, -2, 0, 2, 1, 0, -1])
k7 = k7/np.sqrt(12)
k8 = k7.T
k9 = np.array([1, 0, -1, 0, 0, 0, -1, 0, 1])
k9 = k9/np.sqrt(4)
k10 = np.array([1, -2, 1, -2, 4, -2, 1, -2, 1])
k10 = k10/np.sqrt(36)
v2 = np.array([-2, 0, 2, -1, 0, 1, 0, 0, 0])
k10 = k10/np.sqrt(10)
k10 = np.array([-3, 2, 1, 4, 0, -4, 3, 0, -3])
k10 = k10/np.sqrt(64) 
# 计算平面的基
basis = np.array([k2, v2, k4])


# 定义你的640个9维向量
vectors = np.load('vectors_sin_1920_15_3.npy')  # 请将...替换为你的640个9维向量

# 计算投影
projections = np.dot(vectors, basis.T).reshape(1920, 3)

# 此时，projections是一个640x2的矩阵，每一行是一个原始向量在k1,k2张成的平面上的投影

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制投影结果
ax.scatter(projections[:, 0], projections[:, 1], projections[:, 2])
ax.set_xlabel('Projection on k1')
ax.set_ylabel('Projection on k2')
ax.set_zlabel('Projection on k3')
plt.title('Projection of vectors on the space spanned by k1, k2 and k3')

"""
# 绘制投影结果二维
plt.figure()
plt.scatter(projections[:, 0], projections[:, 1])
plt.xlabel('Projection on k2')
plt.ylabel('Projection on k4')
plt.title('Projection of vectors on the plane spanned by k2 and k4')
"""
plt.show()
"""

#圆拟合
# 定义计算残差的函数
def calc_R(c, x, y):
    ###计算半径###
    return np.sqrt((x-c[0])**2 + (y-c[1])**2)

def f_2(c, x, y):
    ###计算残差###
    Ri = calc_R(c, x, y)
    return Ri - Ri.mean()

# 初始猜测的圆心
center_estimate = [0, 0]
center, ier = leastsq(f_2, center_estimate, args=(projections[:, 0], projections[:, 1]))

# 计算半径
Ri = calc_R(center, projections[:, 0], projections[:, 1])
R = Ri.mean()

# 打印拟合的圆的方程
print(f'The fitted circle equation is: (x-{center[0]:.2f})^2 + (y-{center[1]:.2f})^2 = {R:.2f}^2')


#函数拟合
# 使用numpy的polyfit函数拟合轨迹方程
degree = 2  # 你可以根据需要调整这个值
coefficients = np.polyfit(projections[:, 0], projections[:, 1], degree)

# 打印拟合的轨迹方程
equation = 'y = '
for i in range(degree, -1, -1):
    equation += f'{coefficients[i]:.2f}x^{i} + '
equation = equation[:-3]
print(equation)
"""
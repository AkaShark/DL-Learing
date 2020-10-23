# -*- coding:utf-8 -*-
# datetime:2020-10-22 20:58
# author:Sharker

import numpy as np


def nolin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    # 激活函数 sigmoid
    return 1 / (1 + np.exp(-x))


# 输入
X = np.array([[0.35], [0.9]])
# 输出
y = np.array([[0.5]])

# np.random.seed()函数可以保证生成的随机数具有可预测性。 作用： 可以保证随机初始化值相同
np.random.seed(1)

# 权重
W0 = np.array([[0.1, 0.8], [0.4, 0.6]])
W1 = np.array([[0.3, 0.9]])

print('原始权重:\n', W0, '\n\n', W1)

for j in range(100):
    l0 = X
    # 正向传播
    l1 = nolin(np.dot(W0, l0))
    l2 = nolin(np.dot(W1, l1))

    # l2 为预测值 l2_error为误差
    l2_error = y - l2

    # 平方误差函数 costFunc
    Error = 1 / 2.0 * (y - l2) ** 2
    print(Error)


    # 到这看不明白了
    l2_delta = l2_error * nolin(l2, deriv=True)
    l1_error = l2_delta * W1
    l1_delta = l1_error * nolin(l1, deriv=True)
    W1 += l2_delta * l1.T
    W0 += l0.T.dot(l1_delta)

    print(W0, '\n', W1)


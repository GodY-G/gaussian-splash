import numpy as np

# 五个数据点

point = np.array([[1,2],[2,3],[3,2],[2,1],[1,1]])

# 均值
mu = np.mean(point,axis=0)

# 协方差矩阵
centered = point - mu
Sigma = (centered.T @ centered) / len(point)
print(Sigma)


# 验证单位四元数 = 无旋转
q = np.array([1, 0, 0, 0])    # w=1, x=y=z=0

# 代入公式：
R = [[1-0-0,  0-0,   0+0],
     [0+0,   1-0-0,  0-0],
     [0-0,    0+0,   1-0-0]]
# = I ✓

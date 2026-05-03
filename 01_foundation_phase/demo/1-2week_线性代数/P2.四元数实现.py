import numpy as np
from sympy.abc import theta


# 导入名为numpy的库 用来解决矩阵问题

# 函数 四元数转换为旋转矩阵
def quaternion_to_rotation(q):
    # 一 归一化
    # np.linalg.norm(q)用来计算向量的模长
    # 3d旋转中 四元数的长度必须是1
    q = q / np.linalg.norm(q)

    # 二 拆分变量
    # q是包含四个数字的列表 下面将其解包
    w,x,y,z = q

    # 三 核心公式
    # 创建3x3的矩阵
    #np.array([[],[],[]])表示创建一个三行三列的表格
    res = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],  # 第一行
        [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],  # 第二行
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]  # 第三行
    ])
    return res#返回结果

# -----测试-----


# 1.定义一个绕着z轴旋转的四元数
# np.pi 是圆周率 np.sin np.cos是三角函数
theta = np.pi/4 # 旋转45度
q_test = np.array([np.cos(theta), 0, 0, np.sin(theta)])

# 2.调用函数进行转换

R = quaternion_to_rotation(q_test)

# 3.打印结果
print('旋转矩阵R是：', R)

# 4.验证结果

# 我们拿一个指向 X 轴的向量 [1, 0, 0]。
v = np.array([1, 0, 0])
# R @ v 表示让矩阵 R 作用在向量 v 上（矩阵乘法）。
# 理论上，X轴向量绕Z轴转90度，应该指向Y轴，即得到 [0, 1, 0]。
v_rotated = R @ v
print("旋转后的向量：", v_rotated)

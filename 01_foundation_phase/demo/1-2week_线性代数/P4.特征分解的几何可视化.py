import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 设置打印精度，方便观察
np.set_printoptions(precision=4, suppress=True)

# ====================================================
# 命题 1：对称矩阵的特征向量互相垂直
# ====================================================
print("--- 命题 1 验证 ---")
A = np.array([[3, 1], [1, 2]])
# np.linalg.eigh 专门用于对称矩阵的特征值分解
eigenvalues, eigenvectors = np.linalg.eigh(A)

v1 = eigenvectors[:, 0]  # 第一个特征向量
v2 = eigenvectors[:, 1]  # 第二个特征向量

dot_product = np.dot(v1, v2)
print(f"矩阵 A:\n{A}")
print(f"特征向量 v1: {v1}")
print(f"特征向量 v2: {v2}")
print(f"v1 和 v2 的点积: {dot_product:.10f} (接近0即为垂直)")


# ====================================================
# 命题 2 & 3：旋转不改变特征值，且特征值等于 s²
# ====================================================
def get_rotation_matrix(angle_deg):
    """根据角度返回 2x2 旋转矩阵"""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def plot_ellipse(ax, Sigma, center=(0, 0), label=""):
    """
    根据协方差矩阵 Sigma 画出椭圆
    """
    # 1. 对 Sigma 进行特征值分解
    vals, vecs = np.linalg.eigh(Sigma)
    # 2. 椭圆的轴长是特征值的平方根（标准差）
    # 我们画 2 倍标准差范围，看得更清楚
    width, height = 2 * np.sqrt(vals[0]) * 2, 2 * np.sqrt(vals[1]) * 2
    # 3. 椭圆的旋转角度
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    ell = Ellipse(xy=center, width=width, height=height, angle=angle,
                  edgecolor='blue', fc='None', lw=2, label=label)
    ax.add_patch(ell)

    # 画出特征向量的方向（主轴）
    ax.quiver(0, 0, vecs[0, 0] * np.sqrt(vals[0]), vecs[1, 0] * np.sqrt(vals[0]), scale=1, scale_units='xy', color='r')
    ax.quiver(0, 0, vecs[0, 1] * np.sqrt(vals[1]), vecs[1, 1] * np.sqrt(vals[1]), scale=1, scale_units='xy', color='g')


# 验证准备
s = np.array([2.0, 1.0])  # 缩放因子
S = np.diag(s)  # 缩放矩阵
angles = [0, 30, 60, 90]

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
print("\n--- 命题 2 & 3 验证 ---")

for i, angle in enumerate(angles):
    R = get_rotation_matrix(angle)
    # 构造 Sigma = R * S * S.T * R.T
    Sigma = R @ S @ S.T @ R.T

    # 计算特征值
    eigenvals = np.linalg.eigh(Sigma)[0]

    # 验证特征值是否等于 s^2 (即 2^2=4, 1^2=1)
    print(f"旋转角度 {angle}°: 特征值 = {eigenvals}, s^2 = {s ** 2}")

    # 画图
    plot_ellipse(axs[i], Sigma, label=f"Angle {angle}°")
    axs[i].set_title(f"Rotation: {angle}°\nEig: {eigenvals}")
    axs[i].set_xlim(-5, 5)
    axs[i].set_ylim(-5, 5)
    axs[i].set_aspect('equal')
    axs[i].grid(True)

plt.tight_layout()
plt.show()
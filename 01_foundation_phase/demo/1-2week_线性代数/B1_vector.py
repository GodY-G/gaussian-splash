import numpy as np

a = np.array([2, -1, 3])
b = np.array([1, 4, -2])

# 1. 点积（内积）
dot_product = np.dot(a, b)
print("点积 a·b =", dot_product)
# 判断夹角是锐角还是钝角
if dot_product > 0:
    print("两向量夹角为锐角")
elif dot_product < 0:
    print("两向量夹角为钝角")
else:
    print("两向量垂直")

# 2. 叉积（外积）
cross_product = np.cross(a, b)
print("叉积 a×b =", cross_product)
# 验证是否与 a、b 都垂直（点积为0）
print("叉积与a的点积：", np.dot(cross_product, a))
print("叉积与b的点积：", np.dot(cross_product, b))

# 3. a 在 b 方向上的投影向量
proj_scalar = dot_product / np.linalg.norm(b) ** 2  # 投影标量
proj_vector = proj_scalar * b  # 投影向量
print("a 在 b 方向上的投影向量 =", proj_vector)

# 4. 计算夹角（反余弦）
cos_theta = dot_product / (np.linalg.norm(a) * np.linalg.norm(b))
theta_rad = np.arccos(cos_theta)
theta_deg = np.degrees(theta_rad)
print("两向量夹角（弧度）：", theta_rad)
print("两向量夹角（角度）：", theta_deg)
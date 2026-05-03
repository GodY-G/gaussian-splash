import numpy as np

# 相机参数
fx, fy = 400, 400
cx, cy = 320, 240
angle_y = np.radians(30)  # 绕Y轴旋转30°
C = np.array([2, 0, 5])   # 相机世界坐标位置

# R  t
R = np.array([
    [np.cos(angle_y), 0, np.sin(angle_y)],
    [0, 1, 0],
    [-np.sin(angle_y), 0, np.cos(angle_y)]
])
t = -R @ C
print("旋转矩阵 R：\n", R)
print("平移向量 t：", t)

# 世界坐标变换到相机坐标
P_world = np.array([0,0,0])
P_cam = R @ (P_world - C)
print("世界点的相机坐标是：", P_cam)

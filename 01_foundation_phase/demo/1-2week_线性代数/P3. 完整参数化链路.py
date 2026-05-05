import numpy as np

theta = np.pi/4 # 旋转45度
q = np.array([np.cos(theta), 0, 0, np.sin(theta)])
S = np.diag([2.0, 0.5, 1.0])

def quaternion_to_rotation(q):
    q = q / np.linalg.norm(q)
    w,x,y,z = q
    return np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x**2 + y**2)]
    ])

def gaussian_3d_to_2d(q, s, mu_world, K, R_cam, t_cam):
    #  1. 构建3D协方差矩阵Sigma
    # 决定了高斯球原本的形状
    R_3D = quaternion_to_rotation(q)
    S = np.diag(s)
    L = R_3D @ S
    Sigma_3D  =  L @ L.T

    # 2. 将均值变换到相机坐标系
    # mu_cam = R @ MU_WORLD + t_cam
    mu_cam = R_cam @ mu_world + t_cam
    x, y, z = mu_cam
    depth = z
    # 如果点在相机背面，直接返回空
    if depth <= 0:
        return None, None, depth

    # 3. 计算投影变换的雅可比矩阵 J
    # 这是最难的一步！由于透视投影（近大远小）不是线性的，
    # 我们需要用雅可比矩阵在 mu_cam 这一点做“线性近似”。
    # K[0,0] 是焦距 fx，K[1,1] 是焦距 fy
    fx = K[0, 0]
    fy = K[1, 1]
    # 雅可比矩阵 J 的 2x3 结构
    # 它描述了：相机坐标系下移动 1 毫米，屏幕像素坐标动多少
    J = np.array([
        [fx / z, 0, -(fx * x) / (z ** 2)],
        [0, fy / z, -(fy * y) / (z ** 2)]
    ])

    # 4. 变换协方差矩阵到相机空间，并投影到 2D
    # 4.1 先把 3D 协方差转到相机朝向：Sigma_cam = R_cam * Sigma_3d * R_cam^T
    Sigma_cam = R_cam @ Sigma_3D @ R_cam.T

    # 4.2 再用雅可比矩阵投影到屏幕：Sigma_2d = J * Sigma_cam * J^T
    Sigma_2d = J @ Sigma_cam @ J.T

    # 3DGS 特色优化：为了防止高斯片太小导致渲染闪烁，通常会在对角线加个 0.3 像素的微小偏移
    Sigma_2d[0, 0] += 0.3
    Sigma_2d[1, 1] += 0.3

    # 5. 投影均值（计算像素坐标） ---
    # 使用针孔相机公式：u = fx * (x/z) + cx, v = fy * (y/z) + cy
    cx = K[0, 2]
    cy = K[1, 2]
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    mu_2d = np.array([u, v])

    return mu_2d, Sigma_2d, depth


# --- 测试案例 ---
# 1. 定义相机内参（假设 800x600 分辨率，焦距 500）
K = np.array([[500, 0, 400], [0, 500, 300], [0, 0, 1]])
# 2. 相机外参（假设相机在原点，看着正前方）
R_cam = np.eye(3)
t_cam = np.array([0, 0, 0])
# 3. 3D 高斯参数
q = np.array([1, 0, 0, 0]) # 不旋转
s = np.array([0.2, 0.1, 0.1]) # 细长形状
mu_world = np.array([0, 0, 5]) # 在相机前方 5 米处

# 执行投影
mu_2d, Sigma_2d, depth = gaussian_3d_to_2d(q, s, mu_world, K, R_cam, t_cam)

print(f"2D 像素位置: {mu_2d}")
print(f"2D 协方差矩阵（决定了屏幕上色块的形状）:\n{Sigma_2d}")
print(f"深度值: {depth}")
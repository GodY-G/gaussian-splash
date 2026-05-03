"""
Demo 04: 高斯投影到2D（3DGS完整渲染管线）
==========================================
演示内容：
  1. 3D高斯 → 2D投影（雅可比近似）
  2. 完整的单像素渲染过程
  3. 简化的3DGS渲染管线

运行：python demo_04_高斯投影到2D.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def rotation_y(theta_deg):
    theta = np.radians(theta_deg)
    return np.array([[ np.cos(theta), 0, np.sin(theta)],
                     [ 0,             1, 0            ],
                     [-np.sin(theta), 0, np.cos(theta)]])


def project_gaussian(mu_3d, Sigma_3d, K, R_cam, t_cam):
    """
    将3D高斯投影到2D图像平面

    步骤：
    1. 世界坐标 → 相机坐标
    2. 计算雅可比矩阵 J
    3. 投影协方差：Σ' = J·W·Σ·Wᵀ·Jᵀ（W是相机旋转）
    4. 投影均值：透视除法

    返回：mu_2d, Sigma_2d
    """
    fx, fy = K[0, 0], K[1, 1]

    # 步骤1：世界坐标 → 相机坐标
    mu_cam = R_cam @ mu_3d + t_cam

    # 步骤2：雅可比矩阵（透视投影在μ处的线性化）
    x, y, z = mu_cam
    J = np.array([
        [fx/z,    0,   -fx*x/z**2],
        [0,    fy/z,   -fy*y/z**2],
    ])

    # 步骤3：投影协方差 Σ' = J·W·Σ·Wᵀ·Jᵀ
    W = R_cam  # 相机旋转矩阵
    Sigma_2d = J @ W @ Sigma_3d @ W.T @ J.T

    # 步骤4：投影均值（透视除法）
    mu_2d = np.array([fx * x/z + K[0, 2],
                      fy * y/z + K[1, 2]])

    return mu_2d, Sigma_2d, mu_cam[2]  # 返回深度用于排序


def gaussian_2d_value(x, y, mu, Sigma):
    """计算2D高斯在点(x,y)处的值"""
    pos = np.array([x, y]) - mu
    Sigma_inv = np.linalg.inv(Sigma)
    exponent = -0.5 * pos @ Sigma_inv @ pos
    return np.exp(exponent)  # 不归一化，最大值为1


def render_scene(gaussians, K, R_cam, t_cam, W=320, H=240):
    """
    简化的3DGS渲染管线

    gaussians: list of (mu_3d, Sigma_3d, color, opacity)
    返回: 渲染图像 (H×W×3)
    """
    # 步骤1：投影所有高斯到2D
    projected = []
    for mu_3d, Sigma_3d, color, opacity in gaussians:
        mu_2d, Sigma_2d, depth = project_gaussian(mu_3d, Sigma_3d, K, R_cam, t_cam)
        if depth > 0:  # 只保留在相机前面的高斯
            projected.append((depth, mu_2d, Sigma_2d, color, opacity))

    # 步骤2：按深度排序（从近到远）
    projected.sort(key=lambda x: x[0])

    # 步骤3：对每个像素做alpha合成
    image = np.zeros((H, W, 3))
    T_map = np.ones((H, W))  # 透射率图

    for depth, mu_2d, Sigma_2d, color, base_opacity in projected:
        # 计算这个高斯对每个像素的贡献
        # 只处理高斯影响范围内的像素（3σ范围）
        eigenvalues = np.linalg.eigvalsh(Sigma_2d)
        radius = int(3 * np.sqrt(eigenvalues.max())) + 1

        u_center, v_center = int(mu_2d[0]), int(mu_2d[1])
        u_min = max(0, u_center - radius)
        u_max = min(W, u_center + radius)
        v_min = max(0, v_center - radius)
        v_max = min(H, v_center + radius)

        for v in range(v_min, v_max):
            for u in range(u_min, u_max):
                g_val = gaussian_2d_value(u, v, mu_2d, Sigma_2d)
                alpha = base_opacity * g_val

                if alpha > 0.001:
                    T = T_map[v, u]
                    image[v, u] += T * alpha * np.array(color)
                    T_map[v, u] *= (1 - alpha)

    return np.clip(image, 0, 1)


def demo_single_gaussian_projection():
    """演示单个3D高斯的投影过程"""
    print("="*60)
    print("3D高斯投影到2D：完整过程")
    print("="*60)

    # 相机参数
    W, H = 320, 240
    K = np.array([[300, 0, W/2], [0, 300, H/2], [0, 0, 1]], dtype=float)
    R_cam = np.eye(3)
    t_cam = np.array([0, 0, 5])

    # 3D高斯参数
    mu_3d = np.array([0.3, 0.1, 0.0])  # 世界坐标
    R_gauss = rotation_y(30)
    S = np.diag([1.5, 0.3, 0.5])
    Sigma_3d = R_gauss @ S @ S.T @ R_gauss.T

    print(f"\n3D高斯均值: {mu_3d}")
    print(f"3D协方差矩阵:\n{np.round(Sigma_3d, 4)}")

    mu_2d, Sigma_2d, depth = project_gaussian(mu_3d, Sigma_3d, K, R_cam, t_cam)
    print(f"\n投影后2D均值: {np.round(mu_2d, 2)} 像素")
    print(f"投影后2D协方差:\n{np.round(Sigma_2d, 4)}")
    print(f"深度: {depth:.3f}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("3D高斯投影到2D\nΣ' = J·W·Σ·Wᵀ·Jᵀ", fontsize=12, fontweight='bold')

    # 3D椭球体
    ax3d = fig.add_subplot(121, projection='3d')
    u = np.linspace(0, 2*np.pi, 25)
    v = np.linspace(0, np.pi, 15)
    sphere = np.array([
        np.outer(np.cos(u), np.sin(v)).ravel(),
        np.outer(np.sin(u), np.sin(v)).ravel(),
        np.outer(np.ones_like(u), np.cos(v)).ravel(),
    ])
    L = np.linalg.cholesky(Sigma_3d)
    ellipsoid = (L @ sphere).T + mu_3d
    ex = ellipsoid[:, 0].reshape(25, 15)
    ey = ellipsoid[:, 1].reshape(25, 15)
    ez = ellipsoid[:, 2].reshape(25, 15)
    ax3d.plot_surface(ex, ey, ez, alpha=0.4, color='steelblue')
    ax3d.scatter(*mu_3d, color='red', s=100)

    # 相机位置
    cam_pos = -R_cam.T @ t_cam
    ax3d.scatter(*cam_pos, color='orange', s=200, marker='^')
    ax3d.text(*cam_pos, '相机', color='orange', fontsize=9)

    ax3d.set_title('3D场景')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')

    # 2D投影结果
    ax2d = axes[1]
    xf = np.linspace(0, W, 200)
    yf = np.linspace(0, H, 200)
    Xf, Yf = np.meshgrid(xf, yf)

    pos = np.stack([Xf, Yf], axis=-1)
    diff = pos - mu_2d
    Sigma_inv = np.linalg.inv(Sigma_2d)
    exponent = -0.5 * np.einsum('...i,ij,...j', diff, Sigma_inv, diff)
    Z = np.exp(exponent)

    ax2d.contourf(Xf, Yf, Z, levels=15, cmap='Blues')
    ax2d.contour(Xf, Yf, Z, levels=8, colors='navy', alpha=0.5)
    ax2d.plot(*mu_2d, 'r+', markersize=15, markeredgewidth=2)

    # 画特征向量（主轴）
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma_2d)
    for lam, vec in zip(eigenvalues, eigenvectors.T):
        scale = 2 * np.sqrt(lam)
        ax2d.annotate('', xy=(mu_2d[0]+scale*vec[0], mu_2d[1]+scale*vec[1]),
                      xytext=mu_2d,
                      arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax2d.set_xlim(0, W)
    ax2d.set_ylim(H, 0)
    ax2d.set_title(f'2D投影结果\n均值=({mu_2d[0]:.1f},{mu_2d[1]:.1f})px')
    ax2d.set_xlabel('u（像素）')
    ax2d.set_ylabel('v（像素）')
    ax2d.set_aspect('equal')
    ax2d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_04a_高斯投影.png', dpi=120, bbox_inches='tight')
    print("\n已保存: output_04a_高斯投影.png")
    plt.show()


def demo_simple_3dgs_render():
    """简化的3DGS渲染管线演示"""
    print("\n" + "="*60)
    print("简化3DGS渲染管线（建筑场景）")
    print("="*60)

    W, H = 160, 120  # 小分辨率，加快渲染
    K = np.array([[150, 0, W/2], [0, 150, H/2], [0, 0, 1]], dtype=float)
    R_cam = np.eye(3)
    t_cam = np.array([0, 0, 6])

    # 创建简单的建筑场景高斯基元
    gaussians = []

    # 墙面（扁平高斯，z方向很薄）
    for x in np.linspace(-2, 2, 5):
        for y in np.linspace(-1, 1, 4):
            mu = np.array([x, y, 0.0])
            S = np.diag([0.5, 0.5, 0.05])
            Sigma = S @ S.T
            color = (0.7 + np.random.randn()*0.05,
                     0.7 + np.random.randn()*0.05,
                     0.7 + np.random.randn()*0.05)
            gaussians.append((mu, Sigma, color, 0.8))

    # 窗户（深色）
    for x in [-1, 0, 1]:
        for y in [-0.3, 0.3]:
            mu = np.array([x, y, 0.01])
            S = np.diag([0.2, 0.15, 0.02])
            Sigma = S @ S.T
            gaussians.append((mu, Sigma, (0.2, 0.3, 0.5), 0.9))

    print(f"高斯基元数量: {len(gaussians)}")
    print("渲染中（可能需要几秒）...")

    image = render_scene(gaussians, K, R_cam, t_cam, W, H)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("简化3DGS渲染结果（建筑场景）", fontsize=12, fontweight='bold')

    axes[0].imshow(image)
    axes[0].set_title(f'渲染图像 ({W}×{H})\n{len(gaussians)}个高斯基元')
    axes[0].axis('off')

    # 高斯基元分布（俯视图）
    ax2 = axes[1]
    for mu, Sigma, color, opacity in gaussians:
        ax2.scatter(mu[0], mu[1], c=[color], s=50, alpha=opacity)
    ax2.set_title('高斯基元分布（俯视图）')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_04b_3DGS渲染结果.png', dpi=120, bbox_inches='tight')
    print("已保存: output_04b_3DGS渲染结果.png")
    plt.show()


if __name__ == '__main__':
    print("Demo 04: 高斯投影到2D（3DGS完整渲染管线）")
    print("=" * 50)

    demo_single_gaussian_projection()
    demo_simple_3dgs_render()

    print("\n全部演示完成！")
    print("\n3DGS渲染管线总结：")
    print("  1. 对每个3D高斯：计算雅可比J，投影协方差 Σ'=JWΣWᵀJᵀ")
    print("  2. 按深度排序所有高斯")
    print("  3. 对每个像素：计算各高斯的2D高斯值，做alpha合成")
    print("  4. 整个过程可微分 → 支持梯度下降优化")

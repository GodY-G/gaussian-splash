"""
Demo 04: 协方差矩阵与高斯分布
==============================
演示内容：
  1. 从数据点计算协方差矩阵
  2. 改变 Σ 观察2D高斯椭圆形状变化
  3. 手推 Σ = RSSᵀRᵀ 并可视化（3DGS核心参数化）
  4. 3D高斯椭球体可视化

运行：python demo_04_协方差矩阵与高斯.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def rotation_matrix_2d(theta_deg):
    theta = np.radians(theta_deg)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def gaussian_2d(x, y, mu, Sigma):
    """计算2D高斯概率密度"""
    pos = np.stack([x, y], axis=-1)
    diff = pos - mu
    Sigma_inv = np.linalg.inv(Sigma)
    exponent = -0.5 * np.einsum('...i,ij,...j', diff, Sigma_inv, diff)
    norm = 1 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))
    return norm * np.exp(exponent)

def draw_gaussian_contour(ax, mu, Sigma, color='blue', label='', n_levels=5):
    """画2D高斯的等高线"""
    x = np.linspace(mu[0]-4, mu[0]+4, 200)
    y = np.linspace(mu[1]-4, mu[1]+4, 200)
    X, Y = np.meshgrid(x, y)
    Z = gaussian_2d(X, Y, mu, Sigma)
    ax.contour(X, Y, Z, levels=n_levels, colors=[color], alpha=0.7)
    ax.contourf(X, Y, Z, levels=n_levels, colors=[color], alpha=0.15)
    ax.plot(*mu, 'o', color=color, markersize=6)
    if label:
        ax.text(mu[0]+0.2, mu[1]+0.2, label, color=color, fontsize=9)


# ─────────────────────────────────────────────
# 第一部分：从数据点计算协方差矩阵
# ─────────────────────────────────────────────

def demo_compute_covariance():
    """从散点数据计算协方差矩阵，理解其几何意义"""
    print("="*60)
    print("从数据点计算协方差矩阵")
    print("="*60)

    np.random.seed(42)

    # 生成三种不同分布的数据
    configs = [
        {
            'data': np.random.randn(200, 2),
            'title': '各向同性分布\n（圆形）',
            'color': 'blue'
        },
        {
            'data': np.random.randn(200, 2) * np.array([3, 0.5]),
            'title': '各向异性分布\n（横向拉伸）',
            'color': 'green'
        },
        {
            'data': (rotation_matrix_2d(45) @ (np.random.randn(200, 2) * np.array([3, 0.5])).T).T,
            'title': '旋转分布\n（45°倾斜）',
            'color': 'red'
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("从数据点计算协方差矩阵", fontsize=13, fontweight='bold')

    for ax, cfg in zip(axes, configs):
        data = cfg['data']
        color = cfg['color']

        # 计算均值和协方差
        mu = np.mean(data, axis=0)
        Sigma = np.cov(data.T)  # 2×2 协方差矩阵

        ax.scatter(data[:, 0], data[:, 1], alpha=0.3, s=10, color=color)
        ax.plot(*mu, 'k+', markersize=15, markeredgewidth=2)

        # 画协方差椭圆（1σ, 2σ, 3σ）
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        for n_std, alpha in [(1, 0.5), (2, 0.3), (3, 0.15)]:
            w = 2 * n_std * np.sqrt(eigenvalues[1])
            h = 2 * n_std * np.sqrt(eigenvalues[0])
            ellipse = Ellipse(xy=mu, width=w, height=h, angle=angle,
                              edgecolor=color, facecolor='none', linewidth=2, alpha=alpha+0.3)
            ax.add_patch(ellipse)

        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(cfg['title'], fontsize=10)

        # 显示协方差矩阵
        ax.text(0.02, 0.02,
                f'Σ =\n[[{Sigma[0,0]:.2f}, {Sigma[0,1]:.2f}],\n [{Sigma[1,0]:.2f}, {Sigma[1,1]:.2f}]]',
                transform=ax.transAxes, fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

        print(f"\n{cfg['title'].replace(chr(10), ' ')}:")
        print(f"  均值 μ = {np.round(mu, 3)}")
        print(f"  协方差矩阵 Σ =\n{np.round(Sigma, 3)}")
        print(f"  特征值 = {np.round(eigenvalues, 3)}")

    plt.tight_layout()
    plt.savefig('output_04a_协方差计算.png', dpi=120, bbox_inches='tight')
    print("\n已保存: output_04a_协方差计算.png")
    plt.show()


# ─────────────────────────────────────────────
# 第二部分：3DGS参数化 Σ = RSSᵀRᵀ
# ─────────────────────────────────────────────

def demo_3dgs_covariance_parameterization():
    """
    3DGS的核心参数化：
    不直接优化 Σ（难以保证正定性），
    而是优化 R（旋转）和 S（缩放），
    然后计算 Σ = R S Sᵀ Rᵀ
    """
    print("\n" + "="*60)
    print("3DGS 协方差矩阵参数化：Σ = R S Sᵀ Rᵀ")
    print("="*60)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("3DGS参数化：改变旋转R和缩放S，观察高斯形状变化", fontsize=13, fontweight='bold')

    mu = np.array([0.0, 0.0])

    # 不同的 R 和 S 组合
    configs = [
        (0,   1.0, 1.0, "基准：圆形\nR=0°, S=(1,1)"),
        (0,   3.0, 0.5, "横向拉伸\nR=0°, S=(3,0.5)"),
        (45,  3.0, 0.5, "旋转45°\nR=45°, S=(3,0.5)"),
        (90,  3.0, 0.5, "旋转90°\nR=90°, S=(3,0.5)"),
        (30,  2.0, 1.0, "轻微旋转\nR=30°, S=(2,1)"),
        (60,  2.5, 0.3, "细长旋转\nR=60°, S=(2.5,0.3)"),
        (120, 2.0, 0.8, "另一方向\nR=120°, S=(2,0.8)"),
        (0,   0.5, 2.0, "纵向拉伸\nR=0°, S=(0.5,2)"),
    ]

    for ax, (theta_deg, sx, sy, title) in zip(axes.flat, configs):
        R = rotation_matrix_2d(theta_deg)
        S = np.diag([sx, sy])

        # 核心公式：Σ = R S Sᵀ Rᵀ
        Sigma = R @ S @ S.T @ R.T

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.3)
        ax.axvline(0, color='k', linewidth=0.3)
        ax.set_title(title, fontsize=9)

        draw_gaussian_contour(ax, mu, Sigma, color='steelblue')

        # 画主轴方向（旋转矩阵的列向量）
        e1 = R[:, 0] * sx  # 第一主轴
        e2 = R[:, 1] * sy  # 第二主轴
        ax.annotate('', xy=e1, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.annotate('', xy=e2, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))

        ax.text(0.02, 0.02,
                f'Σ=\n[[{Sigma[0,0]:.1f},{Sigma[0,1]:.1f}],\n [{Sigma[1,0]:.1f},{Sigma[1,1]:.1f}]]',
                transform=ax.transAxes, fontsize=7, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # 验证正定性
        eigenvalues = np.linalg.eigvalsh(Sigma)
        is_pd = np.all(eigenvalues > 0)
        ax.text(0.98, 0.02, f'正定: {"✓" if is_pd else "✗"}',
                transform=ax.transAxes, fontsize=8, ha='right',
                color='green' if is_pd else 'red')

    # 添加图例说明
    fig.text(0.5, 0.01,
             "红箭头 = 第一主轴（R的第一列 × sx）  |  蓝箭头 = 第二主轴（R的第二列 × sy）\n"
             "Σ = R S Sᵀ Rᵀ 天然保证正定性，这是3DGS选择这种参数化的原因",
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightcyan'))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig('output_04b_3DGS参数化.png', dpi=120, bbox_inches='tight')
    print("已保存: output_04b_3DGS参数化.png")
    plt.show()

    # 打印一个具体例子的推导过程
    print("\n具体推导示例（R=45°, S=(3, 0.5)）：")
    R = rotation_matrix_2d(45)
    S = np.diag([3.0, 0.5])
    Sigma = R @ S @ S.T @ R.T
    print(f"R =\n{np.round(R, 4)}")
    print(f"S =\n{S}")
    print(f"S·Sᵀ =\n{S @ S.T}")
    print(f"Σ = R·S·Sᵀ·Rᵀ =\n{np.round(Sigma, 4)}")
    eigenvalues = np.linalg.eigvalsh(Sigma)
    print(f"特征值 = {np.round(eigenvalues, 4)}（均为正数，正定！）")


# ─────────────────────────────────────────────
# 第三部分：3D高斯椭球体可视化
# ─────────────────────────────────────────────

def rotation_matrix_3d(axis, theta_deg):
    """绕指定轴旋转的3D旋转矩阵"""
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:  # z
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def demo_3d_gaussian_ellipsoid():
    """3D高斯椭球体可视化"""
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("3D高斯椭球体（3DGS中每个高斯基元的形状）", fontsize=13, fontweight='bold')

    configs = [
        (np.eye(3),                                    np.diag([1, 1, 1]),   "球形"),
        (rotation_matrix_3d('y', 30),                  np.diag([3, 0.5, 0.5]), "细长椭球（建筑柱子）"),
        (rotation_matrix_3d('x', 45) @ rotation_matrix_3d('z', 30),
                                                        np.diag([2, 1, 0.3]), "扁平椭球（建筑墙面）"),
    ]

    for idx, (R, S, title) in enumerate(configs):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')

        # 计算3D协方差矩阵
        Sigma = R @ S @ S.T @ R.T

        # 生成单位球面上的点
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        sphere = np.stack([x.ravel(), y.ravel(), z.ravel()])

        # 用 Cholesky 分解变换球面到椭球面
        # Σ = L Lᵀ，椭球面 = L · 单位球面
        L = np.linalg.cholesky(Sigma)
        ellipsoid = L @ sphere
        ex = ellipsoid[0].reshape(x.shape)
        ey = ellipsoid[1].reshape(y.shape)
        ez = ellipsoid[2].reshape(z.shape)

        ax.plot_surface(ex, ey, ez, alpha=0.4, color='steelblue', linewidth=0)

        # 画三个主轴
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        colors = ['red', 'green', 'blue']
        labels = ['x轴', 'y轴', 'z轴']
        for i, (lam, vec, color, label) in enumerate(zip(eigenvalues, eigenvectors.T, colors, labels)):
            scale = np.sqrt(lam)
            ax.quiver(0, 0, 0, scale*vec[0], scale*vec[1], scale*vec[2],
                      color=color, arrow_length_ratio=0.15, linewidth=2)

        ax.set_title(f"{title}\nΣ特征值={np.round(eigenvalues, 2)}", fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        lim = np.sqrt(eigenvalues.max()) * 1.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

    plt.tight_layout()
    plt.savefig('output_04c_3D椭球体.png', dpi=120, bbox_inches='tight')
    print("已保存: output_04c_3D椭球体.png")
    plt.show()


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Demo 04: 协方差矩阵与高斯分布")
    print("=" * 50)

    demo_compute_covariance()
    demo_3dgs_covariance_parameterization()
    demo_3d_gaussian_ellipsoid()

    print("\n全部演示完成！")
    print("\n关键结论：")
    print("  1. 协方差矩阵描述数据分布的形状和朝向")
    print("  2. 特征值 = 各主轴方向的方差（轴长²）")
    print("  3. 3DGS用 Σ = RSSᵀRᵀ 参数化，天然保证正定性")
    print("  4. 每个3D高斯基元就是一个椭球体，这是3DGS的核心表示")

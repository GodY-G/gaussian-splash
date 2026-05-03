"""
Demo 03: 多维高斯分布
=====================
演示内容：
  1. 2D高斯分布的等高线与椭圆
  2. 3D高斯椭球体可视化
  3. 3D高斯投影到2D（雅可比近似）

运行：python demo_03_多维高斯分布.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def gaussian_2d(X, Y, mu, Sigma):
    pos = np.stack([X, Y], axis=-1)
    diff = pos - mu
    Sigma_inv = np.linalg.inv(Sigma)
    exponent = -0.5 * np.einsum('...i,ij,...j', diff, Sigma_inv, diff)
    norm = 1 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))
    return norm * np.exp(exponent)


def demo_2d_gaussian():
    """2D高斯分布：等高线与椭圆的关系"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("2D高斯分布：等概率密度线是椭圆", fontsize=13, fontweight='bold')

    mu = np.array([0.0, 0.0])
    configs = [
        (np.array([[1, 0], [0, 1]]),    "各向同性\nΣ=I"),
        (np.array([[3, 0], [0, 1]]),    "x方向拉伸\nσx²=3, σy²=1"),
        (np.array([[1, 0], [0, 3]]),    "y方向拉伸\nσx²=1, σy²=3"),
        (np.array([[2, 1.5], [1.5, 2]]),"正相关\nρ=0.75"),
        (np.array([[2, -1.5], [-1.5, 2]]),"负相关\nρ=-0.75"),
        (np.array([[0.5, 0], [0, 3]]),  "极扁\nσx²=0.5, σy²=3"),
    ]

    xf = np.linspace(-4, 4, 200)
    yf = np.linspace(-4, 4, 200)
    Xf, Yf = np.meshgrid(xf, yf)

    for ax, (Sigma, title) in zip(axes.flat, configs):
        Z = gaussian_2d(Xf, Yf, mu, Sigma)

        ax.contourf(Xf, Yf, Z, levels=15, cmap='Blues')
        ax.contour(Xf, Yf, Z, levels=8, colors='navy', alpha=0.5, linewidths=0.8)

        # 画特征向量（主轴方向）
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        for lam, vec in zip(eigenvalues, eigenvectors.T):
            scale = np.sqrt(lam) * 2
            ax.annotate('', xy=(scale*vec[0], scale*vec[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # 显示协方差矩阵
        ax.text(0.02, 0.02,
                f'Σ=[[{Sigma[0,0]:.1f},{Sigma[0,1]:.1f}],\n   [{Sigma[1,0]:.1f},{Sigma[1,1]:.1f}]]',
                transform=ax.transAxes, fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('output_03a_2D高斯.png', dpi=120, bbox_inches='tight')
    print("已保存: output_03a_2D高斯.png")
    plt.show()


def demo_3d_gaussian_projection():
    """
    3D高斯投影到2D：雅可比近似
    这是3DGS渲染的核心步骤
    """
    print("\n" + "="*60)
    print("3D高斯投影到2D（雅可比近似）")
    print("="*60)

    # 相机内参
    fx, fy = 500.0, 500.0

    # 3D高斯参数
    mu_3d = np.array([0.5, 0.3, 5.0])  # 均值（相机坐标系）
    R = np.array([[np.cos(0.3), -np.sin(0.3), 0],
                  [np.sin(0.3),  np.cos(0.3), 0],
                  [0,            0,            1]])
    S = np.diag([1.5, 0.5, 0.8])
    Sigma_3d = R @ S @ S.T @ R.T

    print(f"\n3D高斯均值（相机坐标）: {mu_3d}")
    print(f"3D协方差矩阵 Σ:\n{np.round(Sigma_3d, 4)}")

    # 投影均值
    z = mu_3d[2]
    mu_2d = np.array([fx * mu_3d[0] / z, fy * mu_3d[1] / z])
    print(f"\n投影后2D均值: {np.round(mu_2d, 4)}")

    # 雅可比矩阵（透视投影在μ处的线性化）
    J = np.array([
        [fx/z,    0,   -fx*mu_3d[0]/z**2],
        [0,    fy/z,   -fy*mu_3d[1]/z**2],
    ])
    print(f"\n雅可比矩阵 J (2×3):\n{np.round(J, 4)}")

    # 2D投影协方差：Σ' = J Σ Jᵀ
    Sigma_2d = J @ Sigma_3d @ J.T
    print(f"\n2D投影协方差 Σ' = J·Σ·Jᵀ:\n{np.round(Sigma_2d, 4)}")

    eigenvalues_2d = np.linalg.eigvalsh(Sigma_2d)
    print(f"2D协方差特征值: {np.round(eigenvalues_2d, 4)}（均为正，正定！）")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("3D高斯投影到2D（雅可比近似）\nΣ' = J·Σ·Jᵀ", fontsize=12, fontweight='bold')

    # 左图：3D椭球体
    ax3d = fig.add_subplot(121, projection='3d')
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sphere = np.array([
        np.outer(np.cos(u), np.sin(v)).ravel(),
        np.outer(np.sin(u), np.sin(v)).ravel(),
        np.outer(np.ones_like(u), np.cos(v)).ravel(),
    ])
    L = np.linalg.cholesky(Sigma_3d)
    ellipsoid = (L @ sphere).T + mu_3d
    ex = ellipsoid[:, 0].reshape(30, 20)
    ey = ellipsoid[:, 1].reshape(30, 20)
    ez = ellipsoid[:, 2].reshape(30, 20)
    ax3d.plot_surface(ex, ey, ez, alpha=0.4, color='steelblue')
    ax3d.scatter(*mu_3d, color='red', s=100, zorder=5)
    ax3d.set_title('3D高斯椭球体')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z（深度）')

    # 右图：2D投影结果
    ax2d = axes[1]
    xf = np.linspace(mu_2d[0]-100, mu_2d[0]+100, 200)
    yf = np.linspace(mu_2d[1]-100, mu_2d[1]+100, 200)
    Xf, Yf = np.meshgrid(xf, yf)
    Z2d = gaussian_2d(Xf, Yf, mu_2d, Sigma_2d)
    ax2d.contourf(Xf, Yf, Z2d, levels=15, cmap='Reds')
    ax2d.contour(Xf, Yf, Z2d, levels=8, colors='darkred', alpha=0.5)
    ax2d.plot(*mu_2d, 'b+', markersize=15, markeredgewidth=2)
    ax2d.set_title('2D投影高斯\n（仍然是高斯！）')
    ax2d.set_xlabel('u（像素）')
    ax2d.set_ylabel('v（像素）')
    ax2d.set_aspect('equal')
    ax2d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_03b_3D投影2D.png', dpi=120, bbox_inches='tight')
    print("\n已保存: output_03b_3D投影2D.png")
    plt.show()


if __name__ == '__main__':
    print("Demo 03: 多维高斯分布")
    print("=" * 50)
    demo_2d_gaussian()
    demo_3d_gaussian_projection()
    print("\n全部演示完成！")

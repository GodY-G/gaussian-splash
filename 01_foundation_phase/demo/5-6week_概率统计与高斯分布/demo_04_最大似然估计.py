"""
Demo 04: 最大似然估计
=====================
演示内容：
  1. 最大似然估计（MLE）原理
  2. 从3D点云拟合高斯分布（3DGS初始化的原理）
  3. 可视化拟合结果

运行：python demo_04_最大似然估计.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def mle_gaussian(data):
    """
    高斯分布的最大似然估计
    MLE解析解：μ_MLE = 样本均值，Σ_MLE = 样本协方差
    """
    mu = np.mean(data, axis=0)
    diff = data - mu
    Sigma = (diff.T @ diff) / len(data)  # 注意：MLE用N，无偏估计用N-1
    return mu, Sigma


def demo_mle_1d():
    """1D高斯的MLE"""
    print("="*60)
    print("最大似然估计（MLE）：从数据估计高斯参数")
    print("="*60)

    np.random.seed(42)
    true_mu, true_sigma = 2.0, 1.5
    n_samples = 100
    data = np.random.normal(true_mu, true_sigma, n_samples)

    # MLE估计
    mu_mle = np.mean(data)
    sigma_mle = np.std(data, ddof=0)  # MLE（除以N）
    sigma_unbiased = np.std(data, ddof=1)  # 无偏估计（除以N-1）

    print(f"\n真实参数: μ={true_mu}, σ={true_sigma}")
    print(f"MLE估计:  μ={mu_mle:.4f}, σ={sigma_mle:.4f}")
    print(f"无偏估计: μ={mu_mle:.4f}, σ={sigma_unbiased:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("1D高斯的最大似然估计", fontsize=12, fontweight='bold')

    x = np.linspace(-3, 7, 300)

    # 左图：数据直方图 + 拟合曲线
    ax = axes[0]
    ax.hist(data, bins=20, density=True, alpha=0.5, color='lightblue', label='观测数据')
    ax.plot(x, stats.norm(true_mu, true_sigma).pdf(x), 'g-', linewidth=2.5,
            label=f'真实分布 N({true_mu},{true_sigma}²)')
    ax.plot(x, stats.norm(mu_mle, sigma_mle).pdf(x), 'r--', linewidth=2.5,
            label=f'MLE拟合 N({mu_mle:.2f},{sigma_mle:.2f}²)')
    ax.set_title(f'数据直方图与拟合曲线\n(n={n_samples}个样本)')
    ax.set_xlabel('x')
    ax.set_ylabel('密度')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 右图：样本量对估计精度的影响
    ax2 = axes[1]
    sample_sizes = [10, 20, 50, 100, 200, 500, 1000]
    mu_errors, sigma_errors = [], []
    for n in sample_sizes:
        errors_mu, errors_sigma = [], []
        for _ in range(100):
            d = np.random.normal(true_mu, true_sigma, n)
            errors_mu.append(abs(np.mean(d) - true_mu))
            errors_sigma.append(abs(np.std(d) - true_sigma))
        mu_errors.append(np.mean(errors_mu))
        sigma_errors.append(np.mean(errors_sigma))

    ax2.loglog(sample_sizes, mu_errors, 'b-o', linewidth=2, label='|μ_MLE - μ_真实|')
    ax2.loglog(sample_sizes, sigma_errors, 'r-s', linewidth=2, label='|σ_MLE - σ_真实|')
    ax2.set_title('样本量对MLE精度的影响\n（样本越多，估计越准）')
    ax2.set_xlabel('样本数量（对数坐标）')
    ax2.set_ylabel('估计误差（对数坐标）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_04a_MLE_1D.png', dpi=120, bbox_inches='tight')
    print("\n已保存: output_04a_MLE_1D.png")
    plt.show()


def demo_mle_3d_pointcloud():
    """
    从3D点云拟合高斯分布
    这模拟了3DGS初始化的过程：
    COLMAP输出稀疏点云 → 用MLE初始化高斯基元
    """
    print("\n" + "="*60)
    print("从3D点云拟合高斯分布（模拟3DGS初始化）")
    print("="*60)

    np.random.seed(42)

    # 模拟建筑场景的稀疏点云（COLMAP输出）
    # 建筑物：主要是平面结构
    def make_building_pointcloud():
        points = []
        # 正面墙（z≈0平面，x和y方向分布广）
        wall_front = np.random.randn(200, 3) * np.array([3, 2, 0.1])
        points.append(wall_front)
        # 侧面墙（x≈3平面）
        wall_side = np.random.randn(100, 3) * np.array([0.1, 2, 1.5]) + np.array([3, 0, 0])
        points.append(wall_side)
        # 屋顶（y≈2平面）
        roof = np.random.randn(80, 3) * np.array([3, 0.1, 1.5]) + np.array([0, 2, 0])
        points.append(roof)
        return np.vstack(points)

    pointcloud = make_building_pointcloud()
    print(f"点云总点数: {len(pointcloud)}")

    # 对整个点云做MLE
    mu_global, Sigma_global = mle_gaussian(pointcloud)
    print(f"\n全局高斯拟合:")
    print(f"  均值 μ = {np.round(mu_global, 3)}")
    print(f"  协方差 Σ =\n{np.round(Sigma_global, 3)}")
    eigenvalues = np.linalg.eigvalsh(Sigma_global)
    print(f"  特征值 = {np.round(eigenvalues, 3)}")
    print(f"  → 最大轴长: {np.sqrt(eigenvalues[-1]):.2f}，最小轴长: {np.sqrt(eigenvalues[0]):.2f}")

    # 可视化
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("从3D点云拟合高斯分布（模拟3DGS初始化）", fontsize=12, fontweight='bold')

    # 3D点云
    ax3d = fig.add_subplot(131, projection='3d')
    ax3d.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2],
                 c='lightblue', s=5, alpha=0.5)
    ax3d.scatter(*mu_global, color='red', s=200, zorder=5, label='MLE均值')

    # 画拟合椭球体
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 15)
    sphere = np.array([
        np.outer(np.cos(u), np.sin(v)).ravel(),
        np.outer(np.sin(u), np.sin(v)).ravel(),
        np.outer(np.ones_like(u), np.cos(v)).ravel(),
    ])
    L = np.linalg.cholesky(Sigma_global)
    ellipsoid = (L @ sphere).T + mu_global
    ex = ellipsoid[:, 0].reshape(20, 15)
    ey = ellipsoid[:, 1].reshape(20, 15)
    ez = ellipsoid[:, 2].reshape(20, 15)
    ax3d.plot_surface(ex, ey, ez, alpha=0.2, color='red')

    ax3d.set_title('3D点云 + 拟合高斯椭球体')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.legend()

    # XY投影
    ax_xy = fig.add_subplot(132)
    ax_xy.scatter(pointcloud[:, 0], pointcloud[:, 1], c='lightblue', s=5, alpha=0.5)
    ax_xy.plot(*mu_global[:2], 'r+', markersize=15, markeredgewidth=2)
    from matplotlib.patches import Ellipse
    eigenvalues_2d, eigenvectors_2d = np.linalg.eigh(Sigma_global[:2, :2])
    angle = np.degrees(np.arctan2(eigenvectors_2d[1, 1], eigenvectors_2d[0, 1]))
    for n_std in [1, 2]:
        e = Ellipse(xy=mu_global[:2],
                    width=2*n_std*np.sqrt(eigenvalues_2d[1]),
                    height=2*n_std*np.sqrt(eigenvalues_2d[0]),
                    angle=angle, edgecolor='red', facecolor='none', linewidth=2, alpha=0.7)
        ax_xy.add_patch(e)
    ax_xy.set_title('XY平面投影')
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.3)

    # 特征值分析
    ax_eig = fig.add_subplot(133)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    bars = ax_eig.bar(['λ₁（最大）', 'λ₂（中等）', 'λ₃（最小）'],
                       eigenvalues_sorted, color=['red', 'orange', 'blue'])
    ax_eig.set_title('协方差矩阵特征值\n（决定椭球体三个轴的长度）')
    ax_eig.set_ylabel('特征值')
    for bar, val in zip(bars, eigenvalues_sorted):
        ax_eig.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}\n(轴长={np.sqrt(val):.2f})',
                    ha='center', fontsize=9)
    ax_eig.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('output_04b_点云拟合.png', dpi=120, bbox_inches='tight')
    print("\n已保存: output_04b_点云拟合.png")
    plt.show()


if __name__ == '__main__':
    print("Demo 04: 最大似然估计")
    print("=" * 50)
    demo_mle_1d()
    demo_mle_3d_pointcloud()
    print("\n全部演示完成！")
    print("\n3DGS初始化流程：")
    print("  COLMAP稀疏点云 → 每个点初始化一个高斯基元")
    print("  初始协方差 = 基于邻近点距离估计的各向同性高斯")
    print("  然后通过训练（梯度下降）优化所有参数")

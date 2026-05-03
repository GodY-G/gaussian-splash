"""
Demo 05: 球谐函数 —— 3DGS的颜色表示
=====================================
演示内容：
  1. 球谐函数的定义和可视化（0-3阶）
  2. 用球谐函数表示方向相关的颜色（视角相关外观）
  3. 球谐系数的物理意义
  4. 与官方代码的对应

与官方代码的对应：
  utils/sh_utils.py: eval_sh()
  scene/gaussian_model.py: get_features(), _features_dc, _features_rest

运行：python demo_05_球谐函数.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# 球谐函数实现（与官方代码一致）
# ─────────────────────────────────────────────

# 球谐函数系数（官方代码 utils/sh_utils.py 中的 C0-C4）
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
       0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
C4 = [2.5033429417967046, -1.7701307697799304, 0.9461746957575601,
      -0.6690465435572892, 0.10578554691520431, -0.6690465435572892,
       0.47308734787878004, -1.7701307697799304, 0.6258357354491761]


def eval_sh_basis(degree, dirs):
    """
    计算球谐基函数值
    degree: 最高阶数（0-3）
    dirs: N×3 方向向量（单位向量）
    返回: N×(degree+1)² 球谐基函数值

    对应官方代码：utils/sh_utils.py eval_sh()
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    result = []

    # 0阶（1个基函数）
    result.append(C0 * np.ones_like(x))

    if degree >= 1:
        # 1阶（3个基函数）
        result.append(-C1 * y)
        result.append( C1 * z)
        result.append(-C1 * x)

    if degree >= 2:
        # 2阶（5个基函数）
        result.append( C2[0] * x * y)
        result.append( C2[1] * y * z)
        result.append( C2[2] * (2*z*z - x*x - y*y))
        result.append( C2[3] * x * z)
        result.append( C2[4] * (x*x - y*y))

    if degree >= 3:
        # 3阶（7个基函数）
        result.append( C3[0] * y * (3*x*x - y*y))
        result.append( C3[1] * x * y * z)
        result.append( C3[2] * y * (4*z*z - x*x - y*y))
        result.append( C3[3] * z * (2*z*z - 3*x*x - 3*y*y))
        result.append( C3[4] * x * (4*z*z - x*x - y*y))
        result.append( C3[5] * z * (x*x - y*y))
        result.append( C3[6] * x * (x*x - 3*y*y))

    return np.stack(result, axis=-1)


def eval_sh_color(sh_coeffs, dirs):
    """
    用球谐系数计算给定方向的颜色
    sh_coeffs: (degree+1)² × 3（每个基函数对应RGB三个系数）
    dirs: N×3 方向向量
    返回: N×3 颜色值
    """
    degree = int(np.sqrt(sh_coeffs.shape[0])) - 1
    basis = eval_sh_basis(degree, dirs)  # N × (degree+1)²
    color = basis @ sh_coeffs            # N × 3
    return np.clip(color + 0.5, 0, 1)   # 加0.5是因为SH的DC项偏移


# ─────────────────────────────────────────────
# 第一部分：球谐基函数可视化
# ─────────────────────────────────────────────

def demo_sh_basis_visualization():
    """可视化球谐基函数（0-3阶）"""
    print("="*60)
    print("球谐函数基函数可视化（0-3阶，共16个）")
    print("="*60)

    # 生成球面上的点
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 100)
    THETA, PHI = np.meshgrid(theta, phi)

    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    dirs = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    basis = eval_sh_basis(3, dirs)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("球谐函数基函数（0-3阶）\n正值=红色，负值=蓝色，大小=绝对值",
                 fontsize=12, fontweight='bold')

    sh_names = [
        'Y₀⁰',
        'Y₁⁻¹', 'Y₁⁰', 'Y₁¹',
        'Y₂⁻²', 'Y₂⁻¹', 'Y₂⁰', 'Y₂¹', 'Y₂²',
        'Y₃⁻³', 'Y₃⁻²', 'Y₃⁻¹', 'Y₃⁰', 'Y₃¹', 'Y₃²', 'Y₃³',
    ]

    for i in range(16):
        ax = fig.add_subplot(4, 4, i+1, projection='3d')
        vals = basis[:, i].reshape(PHI.shape)

        # 用球谐值调制球面半径
        r = np.abs(vals)
        x = r * X
        y = r * Y
        z = r * Z

        # 颜色：正值红色，负值蓝色
        colors = np.where(vals > 0, 'red', 'blue')
        colors_flat = colors.ravel()

        # 画球面
        ax.plot_surface(x, y, z, facecolors=np.where(vals[:, :, np.newaxis] > 0,
                        [1, 0.3, 0.3], [0.3, 0.3, 1]),
                        alpha=0.7, linewidth=0)

        ax.set_title(sh_names[i], fontsize=10)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('output_05a_球谐基函数.png', dpi=100, bbox_inches='tight')
    print("已保存: output_05a_球谐基函数.png")
    plt.show()


# ─────────────────────────────────────────────
# 第二部分：球谐函数表示视角相关颜色
# ─────────────────────────────────────────────

def demo_sh_view_dependent_color():
    """
    演示球谐函数如何表示视角相关的颜色
    这是3DGS中每个高斯基元的颜色表示方式
    """
    print("\n" + "="*60)
    print("球谐函数表示视角相关颜色")
    print("="*60)

    # 生成球面方向
    theta = np.linspace(0, np.pi, 30)
    phi = np.linspace(0, 2*np.pi, 60)
    THETA, PHI = np.meshgrid(theta, phi)
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)
    dirs = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # 三种不同的球谐系数（模拟不同材质）
    np.random.seed(42)
    n_coeffs = 16  # 3阶球谐，16个系数

    configs = [
        {
            'name': '漫反射（各向同性）\n只有DC项非零',
            'coeffs': np.array([[0.5, 0.4, 0.3]] + [[0,0,0]]*15),
        },
        {
            'name': '镜面反射（强方向性）\n高阶项主导',
            'coeffs': np.vstack([
                [0.3, 0.3, 0.3],  # DC
                np.random.randn(3, 3) * 0.1,  # 1阶
                np.random.randn(5, 3) * 0.3,  # 2阶（强）
                np.random.randn(7, 3) * 0.4,  # 3阶（最强）
            ]),
        },
        {
            'name': '建筑玻璃（蓝色反射）\n特定方向蓝色增强',
            'coeffs': np.vstack([
                [0.4, 0.4, 0.6],  # DC：偏蓝
                [[0, 0, 0.2], [0, 0, 0.1], [0, 0, 0]],  # 1阶：z方向蓝色
                np.random.randn(5, 3) * 0.05,
                np.random.randn(7, 3) * 0.02,
            ]),
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
    fig.suptitle("球谐函数表示视角相关颜色\n（从不同方向看，颜色不同）",
                 fontsize=12, fontweight='bold')

    for ax, cfg in zip(axes, configs):
        colors = eval_sh_color(cfg['coeffs'], dirs)
        colors_rgb = colors.reshape(PHI.shape[0], PHI.shape[1], 3)

        ax.plot_surface(X, Y, Z, facecolors=colors_rgb, alpha=0.9, linewidth=0)
        ax.set_title(cfg['name'], fontsize=9)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('output_05b_视角相关颜色.png', dpi=120, bbox_inches='tight')
    print("已保存: output_05b_视角相关颜色.png")
    plt.show()


# ─────────────────────────────────────────────
# 第三部分：球谐阶数对颜色精度的影响
# ─────────────────────────────────────────────

def demo_sh_degree_comparison():
    """对比不同阶数的球谐函数对颜色的近似精度"""
    print("\n" + "="*60)
    print("球谐阶数对颜色精度的影响")
    print("="*60)

    # 目标：一个复杂的视角相关颜色函数
    def target_color(dirs):
        """目标颜色：复杂的视角相关函数"""
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        r = np.clip(0.5 + 0.4*x + 0.3*x*y, 0, 1)
        g = np.clip(0.3 + 0.3*y + 0.2*z, 0, 1)
        b = np.clip(0.6 + 0.3*z - 0.2*x, 0, 1)
        return np.stack([r, g, b], axis=-1)

    # 生成训练方向
    n_dirs = 1000
    np.random.seed(42)
    dirs_train = np.random.randn(n_dirs, 3)
    dirs_train /= np.linalg.norm(dirs_train, axis=1, keepdims=True)
    target = target_color(dirs_train)

    # 用不同阶数的球谐拟合
    degrees = [0, 1, 2, 3]
    errors = []
    fitted_coeffs = []

    for degree in degrees:
        basis = eval_sh_basis(degree, dirs_train)  # N × (d+1)²
        # 最小二乘拟合：coeffs = (BᵀB)⁻¹Bᵀ target
        coeffs, _, _, _ = np.linalg.lstsq(basis, target - 0.5, rcond=None)
        fitted_coeffs.append(coeffs)

        pred = np.clip(basis @ coeffs + 0.5, 0, 1)
        error = np.mean(np.abs(pred - target))
        errors.append(error)
        print(f"  {degree}阶球谐（{(degree+1)**2}个系数）：平均误差 = {error:.4f}")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("不同阶数球谐函数的颜色近似精度", fontsize=12, fontweight='bold')

    # 生成可视化方向
    theta = np.linspace(0, np.pi, 40)
    phi = np.linspace(0, 2*np.pi, 80)
    THETA, PHI = np.meshgrid(theta, phi)
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)
    dirs_vis = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # 目标颜色
    ax_target = fig.add_subplot(231, projection='3d')
    target_vis = target_color(dirs_vis).reshape(PHI.shape[0], PHI.shape[1], 3)
    ax_target.plot_surface(X, Y, Z, facecolors=target_vis, alpha=0.9, linewidth=0)
    ax_target.set_title('目标颜色（真实）', fontsize=10)
    ax_target.axis('off')

    # 各阶数的近似
    for idx, (degree, coeffs) in enumerate(zip(degrees, fitted_coeffs)):
        ax = fig.add_subplot(2, 3, idx+2, projection='3d')
        basis_vis = eval_sh_basis(degree, dirs_vis)
        pred_vis = np.clip(basis_vis @ coeffs + 0.5, 0, 1)
        pred_vis = pred_vis.reshape(PHI.shape[0], PHI.shape[1], 3)
        ax.plot_surface(X, Y, Z, facecolors=pred_vis, alpha=0.9, linewidth=0)
        ax.set_title(f'{degree}阶SH（{(degree+1)**2}系数）\n误差={errors[idx]:.4f}', fontsize=9)
        ax.axis('off')

    # 误差对比
    ax_err = fig.add_subplot(236)
    n_coeffs_list = [(d+1)**2 for d in degrees]
    ax_err.bar(range(len(degrees)), errors, color=['red', 'orange', 'green', 'blue'])
    ax_err.set_xticks(range(len(degrees)))
    ax_err.set_xticklabels([f'{d}阶\n({n}系数)' for d, n in zip(degrees, n_coeffs_list)])
    ax_err.set_title('各阶数的近似误差\n（阶数越高，误差越小）')
    ax_err.set_ylabel('平均绝对误差')
    ax_err.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('output_05c_球谐阶数对比.png', dpi=100, bbox_inches='tight')
    print("\n已保存: output_05c_球谐阶数对比.png")
    plt.show()

    print("\n3DGS中的球谐设置：")
    print("  - 默认使用3阶球谐（16个系数 × 3通道 = 48个参数/高斯）")
    print("  - 训练初期只用0阶（1个系数），逐渐增加阶数")
    print("  - 官方代码：scene/gaussian_model.py 中的 _features_dc 和 _features_rest")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Demo 05: 球谐函数 —— 3DGS的颜色表示")
    print("=" * 60)

    demo_sh_basis_visualization()
    demo_sh_view_dependent_color()
    demo_sh_degree_comparison()

    print("\n全部演示完成！")
    print("\n关键结论：")
    print("  1. 球谐函数是定义在球面上的正交基函数")
    print("  2. 3DGS用球谐系数表示每个高斯的视角相关颜色")
    print("  3. 阶数越高，能表示越复杂的颜色变化（但参数更多）")
    print("  4. 官方代码：utils/sh_utils.py eval_sh()")

"""
Demo 03: 特征值分解
===================
演示内容：
  1. 特征向量可视化：矩阵变换前后方向不变的向量
  2. 对称矩阵的特征分解 A = QΛQᵀ
  3. 验证分解结果
  4. 特征值与椭球体形状的关系（3DGS预热）

运行：python demo_03_特征值分解.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# 第一部分：特征向量的几何意义
# ─────────────────────────────────────────────

def demo_eigenvector_geometry():
    """
    可视化：矩阵变换后，特征向量方向不变（只被拉伸）
    非特征向量方向会改变
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("特征向量：矩阵变换后方向不变的向量", fontsize=13, fontweight='bold')

    # 使用一个有明显特征向量的矩阵
    A = np.array([[3, 1],
                  [1, 3]], dtype=float)

    eigenvalues, eigenvectors = np.linalg.eigh(A)  # eigh 用于对称矩阵
    print(f"矩阵 A =\n{A}")
    print(f"特征值: {eigenvalues}")
    print(f"特征向量（列向量）:\n{eigenvectors}")

    # 生成一圈单位向量
    angles = np.linspace(0, 2*np.pi, 360)
    unit_circle = np.array([np.cos(angles), np.sin(angles)])

    # 变换后的向量
    transformed = A @ unit_circle

    for ax, (pts, title) in zip(axes, [
        (unit_circle, "变换前（单位圆）"),
        (transformed, "变换后（A · v）"),
    ]):
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.plot(pts[0], pts[1], 'lightblue', linewidth=1.5, alpha=0.7)

    # 在两个图上都画特征向量
    colors = ['red', 'green']
    for i, (lam, vec, color) in enumerate(zip(eigenvalues, eigenvectors.T, colors)):
        # 变换前
        axes[0].annotate('', xy=vec, xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        axes[0].text(vec[0]+0.1, vec[1]+0.1,
                     f'v{i+1}=({vec[0]:.2f},{vec[1]:.2f})', color=color, fontsize=9)

        # 变换后（特征向量被拉伸λ倍，方向不变）
        transformed_vec = A @ vec
        axes[1].annotate('', xy=transformed_vec, xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        axes[1].text(transformed_vec[0]+0.1, transformed_vec[1]+0.1,
                     f'A·v{i+1}=λ{i+1}·v{i+1}\nλ{i+1}={lam:.1f}', color=color, fontsize=9)

    # 画一个非特征向量，展示方向会改变
    v_other = np.array([1, 0])
    Av_other = A @ v_other
    axes[0].annotate('', xy=v_other, xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='purple', lw=2, linestyle='dashed'))
    axes[0].text(v_other[0]+0.1, v_other[1]-0.3, '普通向量', color='purple', fontsize=9)
    axes[1].annotate('', xy=Av_other, xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='purple', lw=2, linestyle='dashed'))
    axes[1].text(Av_other[0]+0.1, Av_other[1]-0.3, '方向改变了！', color='purple', fontsize=9)

    legend_elements = [
        mpatches.Patch(color='red',    label=f'特征向量1 (λ={eigenvalues[0]:.1f})'),
        mpatches.Patch(color='green',  label=f'特征向量2 (λ={eigenvalues[1]:.1f})'),
        mpatches.Patch(color='purple', label='普通向量（方向会改变）'),
    ]
    axes[1].legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig('output_03a_特征向量几何.png', dpi=120, bbox_inches='tight')
    print("已保存: output_03a_特征向量几何.png")
    plt.show()


# ─────────────────────────────────────────────
# 第二部分：特征分解 A = QΛQᵀ 的验证
# ─────────────────────────────────────────────

def demo_eigendecomposition_verify():
    """验证对称矩阵的特征分解"""
    print("\n" + "="*60)
    print("对称矩阵特征分解验证：A = Q Λ Qᵀ")
    print("="*60)

    # 构造一个对称正定矩阵
    A = np.array([[4, 2, 0],
                  [2, 3, 1],
                  [0, 1, 2]], dtype=float)

    print(f"\nA =\n{A}")
    print(f"\n验证对称性 A = Aᵀ: {np.allclose(A, A.T)}")

    # 特征分解
    eigenvalues, Q = np.linalg.eigh(A)  # eigh 保证特征值为实数，特征向量正交

    Lambda = np.diag(eigenvalues)

    print(f"\n特征值 λ = {np.round(eigenvalues, 4)}")
    print(f"\n特征向量矩阵 Q（每列是一个特征向量）:\n{np.round(Q, 4)}")
    print(f"\n对角矩阵 Λ:\n{np.round(Lambda, 4)}")

    # 验证 A = Q Λ Qᵀ
    A_reconstructed = Q @ Lambda @ Q.T
    print(f"\n重建 Q·Λ·Qᵀ =\n{np.round(A_reconstructed, 4)}")
    print(f"\n验证 A = Q·Λ·Qᵀ: {np.allclose(A, A_reconstructed)}")

    # 验证 Q 是正交矩阵（QᵀQ = I）
    print(f"\n验证 Q 是正交矩阵 QᵀQ = I: {np.allclose(Q.T @ Q, np.eye(3))}")

    # 验证每个特征向量：A·v = λ·v
    print("\n逐一验证 A·vᵢ = λᵢ·vᵢ:")
    for i in range(3):
        v = Q[:, i]
        lam = eigenvalues[i]
        Av = A @ v
        lam_v = lam * v
        print(f"  v{i+1}: A·v = {np.round(Av, 4)}, λ·v = {np.round(lam_v, 4)}, 相等: {np.allclose(Av, lam_v)}")


# ─────────────────────────────────────────────
# 第三部分：特征值与椭球体形状（3DGS预热）
# ─────────────────────────────────────────────

def draw_ellipse_from_cov(ax, cov, center=(0, 0), n_std=2, color='blue', label=''):
    """根据2×2协方差矩阵画椭圆"""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 椭圆的角度（主轴方向）
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    # 椭圆的半轴长度 = n_std × sqrt(特征值)
    width  = 2 * n_std * np.sqrt(eigenvalues[1])
    height = 2 * n_std * np.sqrt(eigenvalues[0])

    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor=color, alpha=0.2, linewidth=2)
    ax.add_patch(ellipse)

    # 画特征向量（主轴方向）
    for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        scale = n_std * np.sqrt(lam)
        ax.annotate('', xy=(center[0] + scale*vec[0], center[1] + scale*vec[1]),
                    xytext=center,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

    ax.text(center[0]+0.1, center[1]+0.1, label, color=color, fontsize=9, fontweight='bold')


def demo_eigenvalues_and_ellipse():
    """展示不同协方差矩阵对应的椭圆形状"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("协方差矩阵的特征值决定椭球体形状（3DGS核心概念）", fontsize=13, fontweight='bold')

    # 不同的协方差矩阵
    configs = [
        (np.array([[1, 0], [0, 1]]),    "球形\nΣ=I（λ₁=λ₂=1）",    "blue"),
        (np.array([[4, 0], [0, 1]]),    "横向拉伸\nλ₁=4, λ₂=1",    "green"),
        (np.array([[1, 0], [0, 4]]),    "纵向拉伸\nλ₁=1, λ₂=4",    "orange"),
        (np.array([[2, 1.5], [1.5, 2]]), "旋转椭圆\n（有相关性）",   "red"),
        (np.array([[3, -2], [-2, 3]]),  "另一方向旋转",              "purple"),
        (np.array([[0.5, 0], [0, 3]]),  "极扁椭圆\nλ₁=0.5, λ₂=3", "brown"),
    ]

    for ax, (cov, title, color) in zip(axes.flat, configs):
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        ax.set_title(f"{title}", fontsize=10)

        draw_ellipse_from_cov(ax, cov, color=color)

        # 显示特征值
        ax.text(0.02, 0.02,
                f'λ₁={eigenvalues[0]:.2f}\nλ₂={eigenvalues[1]:.2f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # 添加说明
    fig.text(0.5, 0.01,
             "箭头方向 = 特征向量（椭圆主轴方向）  |  箭头长度 = √特征值（半轴长度）\n"
             "3DGS中：每个高斯基元的协方差矩阵 Σ = RSSᵀRᵀ，R决定朝向，S决定轴长",
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightcyan'))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('output_03b_特征值与椭圆.png', dpi=120, bbox_inches='tight')
    print("已保存: output_03b_特征值与椭圆.png")
    plt.show()


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Demo 03: 特征值分解")
    print("=" * 50)

    demo_eigenvector_geometry()
    demo_eigendecomposition_verify()
    demo_eigenvalues_and_ellipse()

    print("\n全部演示完成！")

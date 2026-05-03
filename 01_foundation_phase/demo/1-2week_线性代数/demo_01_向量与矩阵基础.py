"""
Demo 01: 向量与矩阵基础
========================
演示内容：
  1. 向量的点积与叉积（几何可视化）
  2. 矩阵乘法 = 线性变换（单位正方形变换动画）
  3. 转置与逆矩阵的验证

运行：python demo_01_向量与矩阵基础.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# 第一部分：向量点积与叉积的几何意义
# ─────────────────────────────────────────────

def demo_dot_product():
    """点积的几何意义：投影"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("向量点积的几何意义", fontsize=14, fontweight='bold')

    # 左图：不同夹角的点积
    ax = axes[0]
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("点积 = |a||b|cos(θ)")

    a = np.array([3, 0])
    vectors_b = [
        (np.array([2, 1]), 'blue',   '夹角小，点积大'),
        (np.array([1, 2]), 'green',  '夹角中等'),
        (np.array([0, 2]), 'orange', '垂直，点积=0'),
    ]

    ax.annotate('', xy=a, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(a[0]/2, -0.2, 'a=(3,0)', color='red', ha='center', fontsize=10)

    for b, color, label in vectors_b:
        dot = np.dot(a, b)
        ax.annotate('', xy=b, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.text(b[0]+0.1, b[1]+0.1,
                f'b={tuple(b)}\na·b={dot:.1f}',
                color=color, fontsize=9)

    # 右图：投影的几何意义
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 4)
    ax2.set_ylim(-0.5, 3)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title("点积 = a 在 b 方向上的投影 × |b|")

    a = np.array([3, 2])
    b = np.array([3, 0])
    b_hat = b / np.linalg.norm(b)

    # 投影点
    proj_len = np.dot(a, b_hat)
    proj = proj_len * b_hat

    ax2.annotate('', xy=a, xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax2.annotate('', xy=b, xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
    ax2.annotate('', xy=proj, xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2.5))

    # 垂直线
    ax2.plot([a[0], proj[0]], [a[1], proj[1]], 'k--', alpha=0.5)

    ax2.text(a[0]+0.1, a[1], f'a={tuple(a)}', color='red', fontsize=10)
    ax2.text(b[0]+0.1, b[1], f'b={tuple(b)}', color='blue', fontsize=10)
    ax2.text(proj[0]/2, -0.3,
             f'投影长度 = a·b̂ = {proj_len:.1f}', color='green', fontsize=10)

    plt.tight_layout()
    plt.savefig('output_01a_点积.png', dpi=120, bbox_inches='tight')
    print("已保存: output_01a_点积.png")
    plt.show()


def demo_cross_product():
    """叉积的几何意义：法向量"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("叉积 a×b = 垂直于两向量的法向量", fontsize=12)

    a = np.array([2, 0, 0])
    b = np.array([0, 2, 0])
    c = np.cross(a, b)  # 应该是 (0, 0, 4)

    origin = np.zeros(3)

    # 画三个向量
    for vec, color, label in [(a, 'red', 'a'), (b, 'blue', 'b'), (c, 'green', 'a×b')]:
        ax.quiver(*origin, *vec, color=color, arrow_length_ratio=0.15, linewidth=2)
        ax.text(*(origin + vec * 1.1), label, color=color, fontsize=12, fontweight='bold')

    # 画平行四边形（a和b张成的面）
    verts = [[origin, a, a+b, b]]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection(verts, alpha=0.2, facecolor='yellow', edgecolor='gray')
    ax.add_collection3d(poly)

    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 3)
    ax.set_zlim(-0.5, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    area = np.linalg.norm(c)
    ax.text2D(0.05, 0.95,
              f'a×b = {c}\n|a×b| = {area:.1f}（平行四边形面积）',
              transform=ax.transAxes, fontsize=10,
              bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig('output_01b_叉积.png', dpi=120, bbox_inches='tight')
    print("已保存: output_01b_叉积.png")
    plt.show()


# ─────────────────────────────────────────────
# 第二部分：矩阵乘法 = 线性变换
# ─────────────────────────────────────────────

def draw_unit_square(ax, M, color, label, alpha=0.4):
    """把单位正方形经过矩阵M变换后画出来"""
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float).T
    transformed = M @ corners
    ax.fill(transformed[0], transformed[1], alpha=alpha, color=color, label=label)
    ax.plot(transformed[0], transformed[1], color=color, linewidth=2)
    # 画变换后的基向量
    e1 = M @ np.array([1, 0])
    e2 = M @ np.array([0, 1])
    ax.annotate('', xy=e1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=e2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(e1[0]+0.05, e1[1]-0.15, f'e₁→({e1[0]:.1f},{e1[1]:.1f})', color='red', fontsize=9)
    ax.text(e2[0]+0.05, e2[1]+0.05, f'e₂→({e2[0]:.1f},{e2[1]:.1f})', color='blue', fontsize=9)


def demo_matrix_as_transform():
    """矩阵乘法的几何意义：线性变换"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("矩阵乘法 = 线性变换（观察单位正方形如何变化）", fontsize=14, fontweight='bold')

    transforms = [
        (np.eye(2),                          "单位矩阵（不变）",    "gray"),
        (np.array([[2, 0], [0, 1]]),          "水平拉伸 ×2",        "blue"),
        (np.array([[1, 0], [0, 2]]),          "垂直拉伸 ×2",        "green"),
        (np.array([[0, -1], [1, 0]]),         "旋转 90°",           "orange"),
        (np.array([[1, 0.5], [0, 1]]),        "水平剪切",           "purple"),
        (np.array([[2, 0.5], [0.3, 1.5]]),   "复合变换",           "red"),
    ]

    for ax, (M, title, color) in zip(axes.flat, transforms):
        ax.set_xlim(-2.5, 3)
        ax.set_ylim(-2.5, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_title(f"{title}\nM = {M.tolist()}", fontsize=9)

        # 原始正方形（灰色）
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
        ax.fill(corners[0], corners[1], alpha=0.2, color='gray')
        ax.plot(corners[0], corners[1], 'gray', linewidth=1, linestyle='--')

        # 变换后的正方形
        draw_unit_square(ax, M, color, title)

        det = np.linalg.det(M)
        ax.text(0.02, 0.02, f'det(M) = {det:.2f}', transform=ax.transAxes,
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig('output_01c_矩阵变换.png', dpi=120, bbox_inches='tight')
    print("已保存: output_01c_矩阵变换.png")
    plt.show()


# ─────────────────────────────────────────────
# 第三部分：转置与逆矩阵的验证
# ─────────────────────────────────────────────

def demo_transpose_inverse():
    """验证转置和逆矩阵的性质"""
    print("\n" + "="*50)
    print("转置与逆矩阵验证")
    print("="*50)

    A = np.array([[2, 1, 0],
                  [1, 3, 1],
                  [0, 1, 2]], dtype=float)

    print(f"\nA =\n{A}")

    # 转置
    AT = A.T
    print(f"\nAᵀ =\n{AT}")
    print(f"\n验证 (Aᵀ)ᵀ = A: {np.allclose(AT.T, A)}")

    # 逆矩阵
    A_inv = np.linalg.inv(A)
    print(f"\nA⁻¹ =\n{np.round(A_inv, 4)}")
    print(f"\n验证 A · A⁻¹ = I: {np.allclose(A @ A_inv, np.eye(3))}")

    # 对于正交矩阵，转置 = 逆
    theta = np.pi / 4  # 45度旋转
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    print(f"\n旋转矩阵 R (45°) =\n{np.round(R, 4)}")
    print(f"验证 Rᵀ = R⁻¹（正交矩阵的特性）: {np.allclose(R.T, np.linalg.inv(R))}")
    print(f"验证 det(R) = 1（旋转不改变面积）: {np.isclose(np.linalg.det(R), 1.0)}")

    # 矩阵乘法的转置：(AB)ᵀ = BᵀAᵀ
    B = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    C = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
    print(f"\n验证 (BC)ᵀ = CᵀBᵀ: {np.allclose((B @ C).T, C.T @ B.T)}")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Demo 01: 向量与矩阵基础")
    print("=" * 50)
    print("将依次展示：")
    print("  1. 向量点积的几何意义（投影）")
    print("  2. 向量叉积的几何意义（法向量）")
    print("  3. 矩阵乘法 = 线性变换")
    print("  4. 转置与逆矩阵验证")
    print("=" * 50)

    demo_dot_product()
    demo_cross_product()
    demo_matrix_as_transform()
    demo_transpose_inverse()

    print("\n全部演示完成！生成的图片保存在当前目录。")

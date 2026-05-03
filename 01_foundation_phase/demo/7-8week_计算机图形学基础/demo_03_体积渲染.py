"""
Demo 03: 体积渲染
=================
演示内容：
  1. NeRF体积渲染方程的离散实现
  2. 透射率与密度的关系
  3. 3DGS vs NeRF渲染方式对比

运行：python demo_03_体积渲染.py
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def volume_rendering(sigmas, colors, deltas):
    """
    NeRF体积渲染（离散版本）
    C = Σᵢ Tᵢ · (1 - exp(-σᵢδᵢ)) · cᵢ
    Tᵢ = exp(-Σⱼ<ᵢ σⱼδⱼ)

    sigmas: N 体积密度
    colors: N×3 颜色
    deltas: N 采样间隔
    """
    alphas = 1 - np.exp(-sigmas * deltas)  # 不透明度

    T = 1.0
    result = np.zeros(3)
    T_history = [1.0]
    weights = []

    for alpha, c in zip(alphas, colors):
        w = T * alpha
        result += w * c
        weights.append(w)
        T *= (1 - alpha)
        T_history.append(T)

    return result, np.array(weights), np.array(T_history)


def demo_volume_rendering_1d():
    """1D体积渲染：沿一条光线的渲染过程"""
    print("="*60)
    print("NeRF体积渲染方程（离散版本）")
    print("="*60)

    # 沿光线的采样点
    t_vals = np.linspace(0, 10, 100)
    deltas = np.diff(t_vals, append=t_vals[-1]+0.1)

    # 模拟场景：两个物体（前景建筑 + 背景天空）
    # 密度分布：在 t=3 和 t=7 处有两个物体
    sigma = (np.exp(-0.5*((t_vals-3)/0.5)**2) * 5 +
             np.exp(-0.5*((t_vals-7)/1.0)**2) * 2)

    # 颜色分布：前景灰色，背景蓝色
    colors = np.zeros((len(t_vals), 3))
    colors[:, :] = [0.5, 0.7, 1.0]  # 背景蓝色
    mask_fg = t_vals < 5
    colors[mask_fg] = [0.7, 0.7, 0.7]  # 前景灰色

    result, weights, T_history = volume_rendering(sigma, colors, deltas)

    print(f"\n渲染结果: RGB = {np.round(result, 4)}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NeRF体积渲染：沿光线积分\nC = Σᵢ Tᵢ·(1-exp(-σᵢδᵢ))·cᵢ",
                 fontsize=13, fontweight='bold')

    # 密度分布
    ax = axes[0, 0]
    ax.plot(t_vals, sigma, 'b-', linewidth=2)
    ax.fill_between(t_vals, sigma, alpha=0.3, color='blue')
    ax.set_title('体积密度 σ(t)\n（物体在哪里）')
    ax.set_xlabel('深度 t')
    ax.set_ylabel('密度 σ')
    ax.grid(True, alpha=0.3)
    ax.axvline(3, color='gray', linestyle='--', alpha=0.5, label='前景物体')
    ax.axvline(7, color='lightblue', linestyle='--', alpha=0.5, label='背景物体')
    ax.legend()

    # 透射率
    ax2 = axes[0, 1]
    ax2.plot(t_vals, T_history[:-1], 'g-', linewidth=2)
    ax2.fill_between(t_vals, T_history[:-1], alpha=0.2, color='green')
    ax2.set_title('透射率 T(t)\n（光线能穿透到深度t的概率）')
    ax2.set_xlabel('深度 t')
    ax2.set_ylabel('透射率 T')
    ax2.grid(True, alpha=0.3)

    # 权重分布（各采样点对最终颜色的贡献）
    ax3 = axes[1, 0]
    ax3.plot(t_vals, weights, 'r-', linewidth=2)
    ax3.fill_between(t_vals, weights, alpha=0.3, color='red')
    ax3.set_title('权重 w(t) = T(t)·α(t)\n（各深度对最终颜色的贡献）')
    ax3.set_xlabel('深度 t')
    ax3.set_ylabel('权重')
    ax3.grid(True, alpha=0.3)

    # 颜色分布
    ax4 = axes[1, 1]
    ax4.scatter(t_vals, np.ones_like(t_vals)*0.5, c=colors, s=20, alpha=0.7)
    ax4.set_title(f'颜色分布（前景灰色，背景蓝色）\n渲染结果: RGB=({result[0]:.2f},{result[1]:.2f},{result[2]:.2f})')
    ax4.set_xlabel('深度 t')
    ax4.set_yticks([])
    result_patch = plt.Rectangle((0, 0.8), 10, 0.15, color=result)
    ax4.add_patch(result_patch)
    ax4.text(5, 0.87, '渲染结果', ha='center', fontsize=10, color='white', fontweight='bold')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('output_03a_体积渲染.png', dpi=120, bbox_inches='tight')
    print("已保存: output_03a_体积渲染.png")
    plt.show()


def demo_nerf_vs_3dgs():
    """NeRF vs 3DGS渲染方式对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle("NeRF vs 3DGS 渲染方式对比", fontsize=13, fontweight='bold')

    # NeRF渲染示意图
    ax1 = axes[0]
    ax1.set_xlim(-1, 8)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    ax1.set_title('NeRF渲染\n（沿光线采样，每点查询神经网络）', fontsize=11)
    ax1.axis('off')

    # 相机
    ax1.annotate('', xy=(0, 0), xytext=(-0.8, 0),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax1.text(-0.9, 0, '相机', ha='right', fontsize=10)

    # 多条光线
    for dy in [-1.5, -0.75, 0, 0.75, 1.5]:
        ax1.plot([0, 7], [0, dy], 'gray', alpha=0.3, linewidth=1)
        # 采样点
        for t in np.linspace(1, 6, 8):
            x = t
            y = dy * t / 7
            ax1.plot(x, y, 'ro', markersize=5, alpha=0.6)

    # 物体（隐式表示）
    circle = plt.Circle((4, 0), 1.5, color='lightblue', alpha=0.3)
    ax1.add_patch(circle)
    ax1.text(4, 0, '隐式场\n（神经网络）', ha='center', va='center', fontsize=9)

    ax1.text(3, -2.5,
             '每像素：沿光线采样64-128个点\n每点：查询神经网络 → 密度+颜色\n渲染：体积积分\n速度：慢（分钟级）',
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # 3DGS渲染示意图
    ax2 = axes[1]
    ax2.set_xlim(-1, 8)
    ax2.set_ylim(-3, 3)
    ax2.set_aspect('equal')
    ax2.set_title('3DGS渲染\n（高斯投影到2D，alpha合成）', fontsize=11)
    ax2.axis('off')

    # 相机
    ax2.annotate('', xy=(0, 0), xytext=(-0.8, 0),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.text(-0.9, 0, '相机', ha='right', fontsize=10)

    # 3D高斯椭球体
    from matplotlib.patches import Ellipse
    gaussians_3d = [
        (3, 0.5, 1.2, 0.4, 30, 'steelblue'),
        (4, -0.8, 0.8, 0.3, -20, 'steelblue'),
        (5, 0.2, 1.0, 0.5, 45, 'steelblue'),
        (3.5, -0.3, 0.6, 0.8, 0, 'steelblue'),
        (4.5, 1.0, 0.9, 0.3, -30, 'steelblue'),
    ]
    for x, y, w, h, angle, color in gaussians_3d:
        e = Ellipse((x, y), w, h, angle=angle, color=color, alpha=0.4)
        ax2.add_patch(e)

    # 投影箭头
    ax2.annotate('', xy=(1.5, 0), xytext=(3, 0),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.text(2.2, 0.2, '投影', color='red', fontsize=9)

    # 2D高斯（投影结果）
    for y_2d, color in [(-1.5, 'steelblue'), (-0.5, 'steelblue'), (0.5, 'steelblue')]:
        e2d = Ellipse((1, y_2d), 0.3, 0.15, color=color, alpha=0.6)
        ax2.add_patch(e2d)

    ax2.text(3, -2.5,
             '步骤1：将3D高斯投影到2D（Σ\'=JΣJᵀ）\n步骤2：按深度排序\n步骤3：Alpha合成\n速度：快（实时）',
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcyan'))

    plt.tight_layout()
    plt.savefig('output_03b_NeRF_vs_3DGS.png', dpi=120, bbox_inches='tight')
    print("已保存: output_03b_NeRF_vs_3DGS.png")
    plt.show()


if __name__ == '__main__':
    print("Demo 03: 体积渲染")
    print("=" * 50)
    demo_volume_rendering_1d()
    demo_nerf_vs_3dgs()
    print("\n全部演示完成！")

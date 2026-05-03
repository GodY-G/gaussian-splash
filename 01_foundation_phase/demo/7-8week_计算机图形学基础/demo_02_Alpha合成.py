"""
Demo 02: Alpha合成
==================
演示内容：
  1. 前向Alpha合成公式实现
  2. 透射率（Transmittance）的累积过程
  3. 3DGS渲染的Alpha合成可视化

运行：python demo_02_Alpha合成.py
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def alpha_compositing(colors, alphas):
    """
    前向Alpha合成
    C = Σᵢ cᵢ · αᵢ · Πⱼ<ᵢ (1 - αⱼ)

    colors: N×3 (RGB)
    alphas: N (不透明度)
    返回: 合成颜色 (3,)
    """
    result = np.zeros(3)
    transmittance = 1.0  # 初始透射率 T₀ = 1

    for i, (c, a) in enumerate(zip(colors, alphas)):
        result += transmittance * a * c
        transmittance *= (1 - a)
        if transmittance < 1e-4:  # 提前终止（透射率接近0）
            break

    return result, transmittance


def demo_alpha_compositing_basics():
    """Alpha合成基础演示"""
    print("="*60)
    print("前向Alpha合成：C = Σᵢ cᵢ · αᵢ · Πⱼ<ᵢ (1-αⱼ)")
    print("="*60)

    # 示例：3个半透明层
    colors = np.array([
        [1.0, 0.0, 0.0],  # 红色
        [0.0, 1.0, 0.0],  # 绿色
        [0.0, 0.0, 1.0],  # 蓝色
    ])
    alphas = np.array([0.5, 0.5, 0.5])

    print("\n3个半透明层（α=0.5）：")
    print("  层1: 红色 α=0.5")
    print("  层2: 绿色 α=0.5")
    print("  层3: 蓝色 α=0.5")

    result, final_T = alpha_compositing(colors, alphas)
    print(f"\n合成结果: RGB = {np.round(result, 4)}")
    print(f"最终透射率（背景可见度）: {final_T:.4f}")

    # 逐步展示
    print("\n逐步计算过程：")
    T = 1.0
    total = np.zeros(3)
    for i, (c, a) in enumerate(zip(colors, alphas)):
        contribution = T * a * c
        total += contribution
        print(f"  层{i+1}: T={T:.4f}, 贡献={np.round(contribution, 4)}, 累计={np.round(total, 4)}")
        T *= (1 - a)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("前向Alpha合成过程", fontsize=13, fontweight='bold')

    # 左图：各层颜色和透明度
    ax = axes[0]
    layer_names = ['层1\n红色 α=0.5', '层2\n绿色 α=0.5', '层3\n蓝色 α=0.5']
    for i, (c, a, name) in enumerate(zip(colors, alphas, layer_names)):
        rect = plt.Rectangle((i, 0), 0.8, 1, color=c, alpha=a)
        ax.add_patch(rect)
        ax.text(i+0.4, 1.1, name, ha='center', fontsize=9)
    ax.set_xlim(-0.1, 3)
    ax.set_ylim(-0.2, 1.5)
    ax.set_title('各层（从前到后）')
    ax.axis('off')

    # 中图：透射率累积
    ax2 = axes[1]
    T_values = [1.0]
    T = 1.0
    for a in alphas:
        T *= (1 - a)
        T_values.append(T)
    ax2.step(range(len(T_values)), T_values, 'b-o', linewidth=2, markersize=8)
    ax2.fill_between(range(len(T_values)), T_values, alpha=0.2, color='blue')
    ax2.set_title('透射率 T 的累积\n（光线穿透前面所有层的概率）')
    ax2.set_xlabel('经过的层数')
    ax2.set_ylabel('透射率 T')
    ax2.set_xticks(range(len(T_values)))
    ax2.set_xticklabels(['初始', '过层1', '过层2', '过层3'])
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    # 右图：合成结果
    ax3 = axes[2]
    result_img = np.ones((100, 100, 3))
    result_img[:, :] = result
    ax3.imshow(result_img)
    ax3.set_title(f'合成结果\nRGB=({result[0]:.2f},{result[1]:.2f},{result[2]:.2f})')
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig('output_02a_Alpha合成基础.png', dpi=120, bbox_inches='tight')
    print("\n已保存: output_02a_Alpha合成基础.png")
    plt.show()


def demo_3dgs_alpha_compositing():
    """
    模拟3DGS的Alpha合成渲染
    对一个像素，按深度排序的高斯基元做alpha合成
    """
    print("\n" + "="*60)
    print("3DGS Alpha合成渲染模拟")
    print("="*60)

    np.random.seed(42)

    # 模拟一个像素上的高斯基元（按深度排序）
    n_gaussians = 20
    depths = np.sort(np.random.uniform(2, 10, n_gaussians))

    # 每个高斯的颜色（模拟建筑场景：灰色墙面、蓝色天空）
    colors = np.zeros((n_gaussians, 3))
    for i, d in enumerate(depths):
        if d < 5:  # 近处：建筑（灰色）
            colors[i] = np.array([0.7, 0.7, 0.7]) + np.random.randn(3) * 0.05
        else:  # 远处：天空（蓝色）
            colors[i] = np.array([0.5, 0.7, 1.0]) + np.random.randn(3) * 0.05
    colors = np.clip(colors, 0, 1)

    # 每个高斯的不透明度（由2D高斯概率密度决定）
    alphas = np.random.uniform(0.1, 0.8, n_gaussians)

    # 逐步合成
    T = 1.0
    T_history = [1.0]
    contributions = []
    cumulative_color = np.zeros(3)
    cumulative_colors = [cumulative_color.copy()]

    for c, a in zip(colors, alphas):
        contrib = T * a * c
        contributions.append(T * a)  # 权重
        cumulative_color += contrib
        T *= (1 - a)
        T_history.append(T)
        cumulative_colors.append(cumulative_color.copy())
        if T < 1e-4:
            break

    final_color = cumulative_color
    print(f"最终像素颜色: RGB = {np.round(final_color, 4)}")
    print(f"有效高斯数量: {len(contributions)}/{n_gaussians}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("3DGS Alpha合成渲染过程（单像素）", fontsize=13, fontweight='bold')

    # 高斯颜色和深度
    ax = axes[0, 0]
    for i, (d, c, a) in enumerate(zip(depths[:len(contributions)], colors, alphas)):
        ax.bar(i, a, color=c, alpha=0.8)
    ax.set_title('各高斯基元（按深度排序）\n颜色=高斯颜色，高度=不透明度α')
    ax.set_xlabel('高斯索引（从近到远）')
    ax.set_ylabel('不透明度 α')
    ax.grid(True, alpha=0.3, axis='y')

    # 透射率衰减
    ax2 = axes[0, 1]
    ax2.plot(T_history, 'b-o', markersize=4, linewidth=2)
    ax2.fill_between(range(len(T_history)), T_history, alpha=0.2, color='blue')
    ax2.set_title('透射率 T 的衰减\n（越来越多的光被前面的高斯吸收）')
    ax2.set_xlabel('经过的高斯数量')
    ax2.set_ylabel('透射率 T')
    ax2.grid(True, alpha=0.3)

    # 各高斯的贡献权重
    ax3 = axes[1, 0]
    ax3.bar(range(len(contributions)), contributions, color='steelblue', alpha=0.7)
    ax3.set_title('各高斯对最终颜色的贡献权重\n= T × α')
    ax3.set_xlabel('高斯索引')
    ax3.set_ylabel('贡献权重 T·α')
    ax3.grid(True, alpha=0.3, axis='y')

    # 颜色累积过程
    ax4 = axes[1, 1]
    cumulative_arr = np.array(cumulative_colors)
    ax4.plot(cumulative_arr[:, 0], 'r-', linewidth=2, label='R')
    ax4.plot(cumulative_arr[:, 1], 'g-', linewidth=2, label='G')
    ax4.plot(cumulative_arr[:, 2], 'b-', linewidth=2, label='B')
    ax4.axhline(final_color[0], color='r', linestyle='--', alpha=0.5)
    ax4.axhline(final_color[1], color='g', linestyle='--', alpha=0.5)
    ax4.axhline(final_color[2], color='b', linestyle='--', alpha=0.5)
    ax4.set_title(f'颜色累积过程\n最终颜色: ({final_color[0]:.2f},{final_color[1]:.2f},{final_color[2]:.2f})')
    ax4.set_xlabel('经过的高斯数量')
    ax4.set_ylabel('颜色值')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_02b_3DGS渲染.png', dpi=120, bbox_inches='tight')
    print("已保存: output_02b_3DGS渲染.png")
    plt.show()


if __name__ == '__main__':
    print("Demo 02: Alpha合成")
    print("=" * 50)
    demo_alpha_compositing_basics()
    demo_3dgs_alpha_compositing()
    print("\n全部演示完成！")
    print("\n关键结论：")
    print("  1. Alpha合成从前到后叠加，每层贡献 = T × α × c")
    print("  2. 透射率T随深度单调递减（光线被逐渐吸收）")
    print("  3. 3DGS的渲染就是对每个像素做这个alpha合成")
    print("  4. 可微分性：alpha合成对所有参数可微，支持梯度下降")

"""
Demo 02: 一维高斯分布
=====================
演示内容：
  1. 参数μ和σ对高斯形状的影响
  2. 68-95-99.7规则可视化
  3. 高斯函数的可微性（梯度可视化）

运行：python demo_02_一维高斯分布.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))


def demo_gaussian_parameters():
    """μ和σ对高斯形状的影响"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("一维高斯分布参数的影响", fontsize=13, fontweight='bold')

    x = np.linspace(-6, 6, 500)

    # 左图：改变μ（位置）
    ax = axes[0]
    for mu, color in [(-2, 'blue'), (0, 'green'), (2, 'red'), (3, 'orange')]:
        y = gaussian(x, mu, 1.0)
        ax.plot(x, y, color=color, linewidth=2, label=f'μ={mu}, σ=1')
        ax.axvline(mu, color=color, linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title('改变μ（均值）：控制中心位置')
    ax.set_xlabel('x')
    ax.set_ylabel('p(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：改变σ（宽度）
    ax2 = axes[1]
    for sigma, color in [(0.5, 'blue'), (1.0, 'green'), (2.0, 'red'), (3.0, 'orange')]:
        y = gaussian(x, 0, sigma)
        ax2.plot(x, y, color=color, linewidth=2, label=f'μ=0, σ={sigma}')
    ax2.set_title('改变σ（标准差）：控制宽窄')
    ax2.set_xlabel('x')
    ax2.set_ylabel('p(x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_02a_高斯参数.png', dpi=120, bbox_inches='tight')
    print("已保存: output_02a_高斯参数.png")
    plt.show()


def demo_68_95_997_rule():
    """68-95-99.7规则可视化"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("68-95-99.7规则：高斯分布的概率集中区间", fontsize=12, fontweight='bold')

    mu, sigma = 0, 1
    x = np.linspace(-4, 4, 500)
    y = gaussian(x, mu, sigma)

    ax.plot(x, y, 'k-', linewidth=2.5)

    # 三个区间的阴影
    intervals = [
        (1, '#3498db', '68.27%：μ±1σ'),
        (2, '#2ecc71', '95.45%：μ±2σ'),
        (3, '#e74c3c', '99.73%：μ±3σ'),
    ]

    for n_sigma, color, label in reversed(intervals):
        x_fill = np.linspace(mu - n_sigma*sigma, mu + n_sigma*sigma, 300)
        y_fill = gaussian(x_fill, mu, sigma)
        prob = stats.norm.cdf(n_sigma) - stats.norm.cdf(-n_sigma)
        ax.fill_between(x_fill, y_fill, alpha=0.3, color=color, label=f'{label} ({prob:.2%})')
        ax.axvline(mu - n_sigma*sigma, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(mu + n_sigma*sigma, color=color, linestyle='--', linewidth=1.5, alpha=0.7)

    # 标注
    for n_sigma in [1, 2, 3]:
        ax.text(mu + n_sigma*sigma + 0.05, gaussian(mu + n_sigma*sigma, mu, sigma) + 0.01,
                f'+{n_sigma}σ', fontsize=10, ha='left')
        ax.text(mu - n_sigma*sigma - 0.05, gaussian(mu - n_sigma*sigma, mu, sigma) + 0.01,
                f'-{n_sigma}σ', fontsize=10, ha='right')

    ax.axvline(mu, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.text(mu + 0.05, ax.get_ylim()[1]*0.95, 'μ', fontsize=12)

    ax.set_xlabel('x（以σ为单位）')
    ax.set_ylabel('概率密度 p(x)')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 添加说明
    ax.text(0.02, 0.6,
            '3DGS中的意义：\n高斯基元的"影响范围"\n通常取3σ以内\n超出范围的贡献可忽略',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig('output_02b_68_95_997规则.png', dpi=120, bbox_inches='tight')
    print("已保存: output_02b_68_95_997规则.png")
    plt.show()


def demo_gaussian_gradient():
    """高斯函数的可微性：梯度可视化"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("高斯函数的可微性（3DGS能做梯度下降的原因）", fontsize=12, fontweight='bold')

    mu, sigma = 0.0, 1.0
    x = np.linspace(-4, 4, 500)
    g = gaussian(x, mu, sigma)

    # 关于μ的梯度
    dg_dmu = g * (x - mu) / sigma**2
    # 关于σ的梯度
    dg_dsigma = g * ((x - mu)**2 / sigma**3 - 1/sigma)

    for ax, (y, title, color) in zip(axes, [
        (g,        'N(x; μ=0, σ=1)\n高斯函数本身',    'blue'),
        (dg_dmu,   '∂N/∂μ\n关于均值的梯度',           'red'),
        (dg_dsigma,'∂N/∂σ\n关于标准差的梯度',         'green'),
    ]):
        ax.plot(x, y, color=color, linewidth=2.5)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(mu, color='gray', linestyle='--', linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x')
        ax.grid(True, alpha=0.3)
        ax.fill_between(x, y, alpha=0.15, color=color)

    axes[1].text(0.05, 0.95,
                 '正值：增大μ会增大N(x)\n负值：增大μ会减小N(x)',
                 transform=axes[1].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow'), va='top')

    plt.tight_layout()
    plt.savefig('output_02c_高斯梯度.png', dpi=120, bbox_inches='tight')
    print("已保存: output_02c_高斯梯度.png")
    plt.show()


if __name__ == '__main__':
    print("Demo 02: 一维高斯分布")
    print("=" * 50)
    demo_gaussian_parameters()
    demo_68_95_997_rule()
    demo_gaussian_gradient()
    print("\n全部演示完成！")

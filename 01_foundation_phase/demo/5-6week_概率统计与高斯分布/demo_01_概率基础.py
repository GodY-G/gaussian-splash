"""
Demo 01: 概率基础
=================
演示内容：
  1. 概率密度函数（PDF）与累积分布函数（CDF）
  2. 期望与方差的几何意义
  3. 常见分布对比

运行：python demo_01_概率基础.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def demo_pdf_cdf():
    """PDF和CDF的关系"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("概率密度函数（PDF）与累积分布函数（CDF）", fontsize=13, fontweight='bold')

    distributions = [
        (stats.norm(0, 1),    '标准正态分布\nN(0,1)',    'blue'),
        (stats.uniform(-2,4), '均匀分布\nU(-2,2)',       'green'),
        (stats.expon(0, 1),   '指数分布\nExp(1)',        'orange'),
    ]

    x = np.linspace(-4, 6, 500)

    for col, (dist, title, color) in enumerate(distributions):
        # PDF
        ax_pdf = axes[0, col]
        pdf = dist.pdf(x)
        ax_pdf.plot(x, pdf, color=color, linewidth=2.5)
        ax_pdf.fill_between(x, pdf, alpha=0.2, color=color)
        ax_pdf.set_title(f'{title}\nPDF', fontsize=10)
        ax_pdf.set_ylabel('概率密度 p(x)')
        ax_pdf.set_xlabel('x')
        ax_pdf.grid(True, alpha=0.3)

        # 标注均值和标准差
        mu = dist.mean()
        sigma = dist.std()
        ax_pdf.axvline(mu, color='red', linestyle='--', linewidth=1.5, label=f'μ={mu:.2f}')
        ax_pdf.axvline(mu-sigma, color='gray', linestyle=':', linewidth=1)
        ax_pdf.axvline(mu+sigma, color='gray', linestyle=':', linewidth=1,
                       label=f'σ={sigma:.2f}')
        ax_pdf.legend(fontsize=8)

        # 阴影：P(-1 < X < 1)
        x_fill = np.linspace(max(x[0], mu-sigma), min(x[-1], mu+sigma), 200)
        ax_pdf.fill_between(x_fill, dist.pdf(x_fill), alpha=0.4, color=color,
                            label=f'P(μ±σ)={dist.cdf(mu+sigma)-dist.cdf(mu-sigma):.2%}')
        ax_pdf.legend(fontsize=8)

        # CDF
        ax_cdf = axes[1, col]
        cdf = dist.cdf(x)
        ax_cdf.plot(x, cdf, color=color, linewidth=2.5)
        ax_cdf.set_title(f'{title}\nCDF', fontsize=10)
        ax_cdf.set_ylabel('累积概率 F(x) = P(X≤x)')
        ax_cdf.set_xlabel('x')
        ax_cdf.grid(True, alpha=0.3)
        ax_cdf.axhline(0.5, color='red', linestyle='--', linewidth=1, label='中位数')
        ax_cdf.axhline(0.25, color='gray', linestyle=':', linewidth=1)
        ax_cdf.axhline(0.75, color='gray', linestyle=':', linewidth=1)
        ax_cdf.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('output_01a_PDF_CDF.png', dpi=120, bbox_inches='tight')
    print("已保存: output_01a_PDF_CDF.png")
    plt.show()


def demo_expectation_variance():
    """期望和方差的几何意义"""
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("期望与方差的几何意义", fontsize=13, fontweight='bold')

    # 左图：期望 = 分布的"重心"
    ax = axes[0]
    x = np.linspace(-5, 5, 500)

    for mu, sigma, color, label in [
        (-2, 1, 'blue',   'N(-2, 1)'),
        ( 0, 1, 'green',  'N(0, 1)'),
        ( 2, 1, 'orange', 'N(2, 1)'),
    ]:
        pdf = stats.norm(mu, sigma).pdf(x)
        ax.plot(x, pdf, color=color, linewidth=2, label=label)
        ax.axvline(mu, color=color, linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_title('期望μ = 分布的中心位置（重心）')
    ax.set_xlabel('x')
    ax.set_ylabel('p(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：方差 = 分布的"宽度"
    ax2 = axes[1]
    for sigma, color, label in [
        (0.5, 'blue',   'N(0, 0.5²)：窄'),
        (1.0, 'green',  'N(0, 1²)：中'),
        (2.0, 'orange', 'N(0, 2²)：宽'),
    ]:
        pdf = stats.norm(0, sigma).pdf(x)
        ax2.plot(x, pdf, color=color, linewidth=2, label=label)
        # 标注±σ范围
        ax2.axvspan(-sigma, sigma, alpha=0.05, color=color)

    ax2.set_title('方差σ² = 分布的宽度（分散程度）')
    ax2.set_xlabel('x')
    ax2.set_ylabel('p(x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_01b_期望方差.png', dpi=120, bbox_inches='tight')
    print("已保存: output_01b_期望方差.png")
    plt.show()


if __name__ == '__main__':
    print("Demo 01: 概率基础")
    print("=" * 50)
    demo_pdf_cdf()
    demo_expectation_variance()
    print("\n全部演示完成！")

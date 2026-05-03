"""
Demo 01: 梯度场可视化
=====================
演示内容：
  1. 梯度向量场（quiver图）
  2. 等高线与梯度的关系（梯度垂直于等高线）
  3. 鞍点、极小值、极大值的梯度特征

运行：python demo_01_梯度场可视化.py
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def demo_gradient_field():
    """可视化不同函数的梯度场"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("梯度场可视化：梯度指向函数值增长最快的方向", fontsize=13, fontweight='bold')

    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)

    # 用于等高线的细网格
    xf = np.linspace(-3, 3, 200)
    yf = np.linspace(-3, 3, 200)
    Xf, Yf = np.meshgrid(xf, yf)

    functions = [
        {
            'f':  lambda x, y: x**2 + y**2,
            'gx': lambda x, y: 2*x,
            'gy': lambda x, y: 2*y,
            'title': 'f(x,y) = x² + y²\n（碗形，唯一极小值）',
            'color': 'blue'
        },
        {
            'f':  lambda x, y: x**2 + 2*y**2,
            'gx': lambda x, y: 2*x,
            'gy': lambda x, y: 4*y,
            'title': 'f(x,y) = x² + 2y²\n（椭圆碗，y方向更陡）',
            'color': 'green'
        },
        {
            'f':  lambda x, y: x**2 - y**2,
            'gx': lambda x, y: 2*x,
            'gy': lambda x, y: -2*y,
            'title': 'f(x,y) = x² - y²\n（鞍点，原点处梯度=0但非极值）',
            'color': 'red'
        },
        {
            'f':  lambda x, y: np.sin(x) * np.cos(y),
            'gx': lambda x, y: np.cos(x) * np.cos(y),
            'gy': lambda x, y: -np.sin(x) * np.sin(y),
            'title': 'f(x,y) = sin(x)cos(y)\n（多个极值，复杂地形）',
            'color': 'purple'
        },
        {
            'f':  lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2,
            'gx': lambda x, y: 2*(x**2+y-11)*2*x + 2*(x+y**2-7),
            'gy': lambda x, y: 2*(x**2+y-11) + 2*(x+y**2-7)*2*y,
            'title': 'Himmelblau函数\n（4个全局极小值，优化难题）',
            'color': 'orange'
        },
        {
            'f':  lambda x, y: 0.5*(x**2 + 10*y**2),
            'gx': lambda x, y: x,
            'gy': lambda x, y: 10*y,
            'title': 'f(x,y) = 0.5(x² + 10y²)\n（病态：y方向梯度远大于x）',
            'color': 'brown'
        },
    ]

    for ax, fn in zip(axes.flat, functions):
        Z = fn['f'](Xf, Yf)
        GX = fn['gx'](X, Y)
        GY = fn['gy'](X, Y)

        # 归一化梯度向量（只显示方向）
        magnitude = np.sqrt(GX**2 + GY**2) + 1e-10
        GX_norm = GX / magnitude
        GY_norm = GY / magnitude

        # 等高线
        try:
            ax.contourf(Xf, Yf, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
            ax.contour(Xf, Yf, Z, levels=20, colors='gray', alpha=0.4, linewidths=0.5)
        except Exception:
            pass

        # 梯度箭头
        ax.quiver(X, Y, GX_norm, GY_norm,
                  magnitude, cmap='hot', alpha=0.8,
                  scale=25, width=0.003)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(fn['title'], fontsize=9)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # 标注梯度为0的点（极值点）
        ax.plot(0, 0, 'w*', markersize=12, zorder=5)

    plt.tight_layout()
    plt.savefig('output_01a_梯度场.png', dpi=120, bbox_inches='tight')
    print("已保存: output_01a_梯度场.png")
    plt.show()


def demo_gradient_perpendicular_to_contour():
    """演示梯度垂直于等高线"""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("梯度垂直于等高线\n（梯度指向等高线法线方向）", fontsize=12, fontweight='bold')

    xf = np.linspace(-3, 3, 300)
    yf = np.linspace(-3, 3, 300)
    Xf, Yf = np.meshgrid(xf, yf)
    Z = Xf**2 + 2*Yf**2

    # 等高线
    cs = ax.contour(Xf, Yf, Z, levels=8, cmap='Blues')
    ax.clabel(cs, inline=True, fontsize=8)

    # 在几个点上画梯度向量
    sample_points = [
        (1, 0.5), (-1, 0.5), (0.5, -1), (-0.5, -1),
        (2, 0.3), (-2, 0.3),
    ]

    for px, py in sample_points:
        gx = 2 * px
        gy = 4 * py
        mag = np.sqrt(gx**2 + gy**2)
        scale = 0.4
        ax.annotate('', xy=(px + scale*gx/mag, py + scale*gy/mag),
                    xytext=(px, py),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.plot(px, py, 'ro', markersize=5)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', label='等高线 f(x,y)=c'),
        Line2D([0], [0], color='red', marker='>', label='梯度方向（垂直于等高线）'),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()
    plt.savefig('output_01b_梯度垂直等高线.png', dpi=120, bbox_inches='tight')
    print("已保存: output_01b_梯度垂直等高线.png")
    plt.show()


if __name__ == '__main__':
    print("Demo 01: 梯度场可视化")
    print("=" * 50)
    demo_gradient_field()
    demo_gradient_perpendicular_to_contour()
    print("\n全部演示完成！")

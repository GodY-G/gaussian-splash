"""
Demo 02: 梯度下降
=================
演示内容：
  1. 梯度下降收敛过程动画
  2. 不同学习率的影响（太大/太小/合适）
  3. 病态函数上的梯度下降（为什么需要Adam）

运行：python demo_02_梯度下降.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# 梯度下降核心实现
# ─────────────────────────────────────────────

def gradient_descent(f, grad_f, x0, lr, n_steps=100):
    """
    标准梯度下降
    f: 目标函数
    grad_f: 梯度函数
    x0: 初始点（numpy数组）
    lr: 学习率
    """
    x = x0.copy().astype(float)
    trajectory = [x.copy()]
    losses = [f(*x)]

    for _ in range(n_steps):
        g = grad_f(*x)
        x = x - lr * g
        trajectory.append(x.copy())
        losses.append(f(*x))

    return np.array(trajectory), np.array(losses)


# ─────────────────────────────────────────────
# 第一部分：不同学习率的影响
# ─────────────────────────────────────────────

def demo_learning_rate_effect():
    """对比不同学习率的收敛行为"""
    # 目标函数：f(x,y) = x² + 2y²
    f     = lambda x, y: x**2 + 2*y**2
    grad_f = lambda x, y: np.array([2*x, 4*y])

    x0 = np.array([2.5, 2.0])
    learning_rates = [0.05, 0.2, 0.45, 0.6]
    lr_labels = ['太小 (0.05)', '合适 (0.2)', '偏大 (0.45)', '太大 (0.6)']
    colors = ['green', 'blue', 'orange', 'red']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("学习率对梯度下降的影响\nf(x,y) = x² + 2y²", fontsize=13, fontweight='bold')

    # 背景等高线
    xf = np.linspace(-3, 3, 200)
    yf = np.linspace(-3, 3, 200)
    Xf, Yf = np.meshgrid(xf, yf)
    Zf = f(Xf, Yf)

    for ax, lr, label, color in zip(axes.flat, learning_rates, lr_labels, colors):
        traj, losses = gradient_descent(f, grad_f, x0, lr, n_steps=50)

        ax.contourf(Xf, Yf, Zf, levels=20, cmap='RdYlBu_r', alpha=0.5)
        ax.contour(Xf, Yf, Zf, levels=20, colors='gray', alpha=0.3, linewidths=0.5)

        # 画轨迹
        ax.plot(traj[:, 0], traj[:, 1], 'o-', color=color, markersize=4,
                linewidth=1.5, label=f'lr={lr}', zorder=5)
        ax.plot(traj[0, 0], traj[0, 1], 'k^', markersize=10, zorder=6, label='起点')
        ax.plot(0, 0, 'w*', markersize=15, zorder=6, label='极小值(0,0)')

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(f"学习率 {label}\n最终损失: {losses[-1]:.6f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # 标注步数
        for i in range(0, min(len(traj), 10)):
            ax.text(traj[i, 0]+0.05, traj[i, 1]+0.05, str(i), fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig('output_02a_学习率影响.png', dpi=120, bbox_inches='tight')
    print("已保存: output_02a_学习率影响.png")
    plt.show()

    # 损失曲线对比
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("不同学习率的损失收敛曲线", fontsize=12, fontweight='bold')

    for lr, label, color in zip(learning_rates, lr_labels, colors):
        _, losses = gradient_descent(f, grad_f, x0, lr, n_steps=100)
        losses_clipped = np.clip(losses, 0, 20)  # 防止发散时图形失真
        ax.semilogy(losses_clipped, color=color, linewidth=2, label=label)

    ax.set_xlabel('迭代步数')
    ax.set_ylabel('损失值（对数坐标）')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_02b_损失曲线.png', dpi=120, bbox_inches='tight')
    print("已保存: output_02b_损失曲线.png")
    plt.show()


# ─────────────────────────────────────────────
# 第二部分：病态函数——为什么需要自适应学习率
# ─────────────────────────────────────────────

def demo_ill_conditioned():
    """
    病态函数：f(x,y) = 0.5(x² + 100y²)
    x方向梯度小，y方向梯度大（相差100倍）
    普通梯度下降会在y方向震荡
    """
    # 病态函数
    f      = lambda x, y: 0.5 * (x**2 + 100*y**2)
    grad_f = lambda x, y: np.array([x, 100*y])

    x0 = np.array([1.5, 0.5])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("病态函数上的梯度下降\nf(x,y) = 0.5(x² + 100y²)，x和y方向梯度相差100倍",
                 fontsize=12, fontweight='bold')

    xf = np.linspace(-2, 2, 200)
    yf = np.linspace(-0.6, 0.6, 200)
    Xf, Yf = np.meshgrid(xf, yf)
    Zf = f(Xf, Yf)

    for ax, (lr, label, color) in zip(axes, [
        (0.009, '学习率=0.009（勉强收敛，但震荡）', 'orange'),
        (0.001, '学习率=0.001（收敛慢，步子太小）', 'blue'),
    ]):
        traj, losses = gradient_descent(f, grad_f, x0, lr, n_steps=200)

        ax.contourf(Xf, Yf, Zf, levels=20, cmap='RdYlBu_r', alpha=0.5)
        ax.contour(Xf, Yf, Zf, levels=20, colors='gray', alpha=0.3, linewidths=0.5)
        ax.plot(traj[:, 0], traj[:, 1], 'o-', color=color, markersize=2,
                linewidth=1, alpha=0.7)
        ax.plot(traj[0, 0], traj[0, 1], 'k^', markersize=10, zorder=6)
        ax.plot(0, 0, 'w*', markersize=15, zorder=6)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.6, 0.6)
        ax.set_title(f"{label}\n200步后损失: {losses[-1]:.6f}", fontsize=9)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig('output_02c_病态函数.png', dpi=120, bbox_inches='tight')
    print("已保存: output_02c_病态函数.png")
    print("\n结论：病态函数上，普通梯度下降效率很低。")
    print("Adam通过自适应学习率解决这个问题（见 demo_03_Adam优化器.py）")
    plt.show()


# ─────────────────────────────────────────────
# 第三部分：梯度下降动画（可选，需要保存gif）
# ─────────────────────────────────────────────

def demo_gradient_descent_animation():
    """梯度下降收敛动画"""
    f      = lambda x, y: x**2 + 2*y**2
    grad_f = lambda x, y: np.array([2*x, 4*y])

    x0 = np.array([2.5, 2.0])
    traj, losses = gradient_descent(f, grad_f, x0, lr=0.2, n_steps=30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("梯度下降动画：f(x,y) = x² + 2y²", fontsize=12, fontweight='bold')

    xf = np.linspace(-3, 3, 200)
    yf = np.linspace(-3, 3, 200)
    Xf, Yf = np.meshgrid(xf, yf)
    Zf = f(Xf, Yf)

    ax1.contourf(Xf, Yf, Zf, levels=20, cmap='RdYlBu_r', alpha=0.6)
    ax1.contour(Xf, Yf, Zf, levels=20, colors='gray', alpha=0.3, linewidths=0.5)
    ax1.plot(0, 0, 'w*', markersize=15, zorder=6)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    ax1.set_title('参数空间轨迹')

    line1, = ax1.plot([], [], 'b-o', markersize=5, linewidth=2)
    point1, = ax1.plot([], [], 'ro', markersize=10, zorder=7)
    step_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='white'))

    ax2.set_xlim(0, len(losses))
    ax2.set_ylim(0, losses[0] * 1.1)
    ax2.set_xlabel('迭代步数')
    ax2.set_ylabel('损失值')
    ax2.set_title('损失曲线')
    ax2.grid(True, alpha=0.3)
    line2, = ax2.plot([], [], 'b-', linewidth=2)

    def animate(i):
        line1.set_data(traj[:i+1, 0], traj[:i+1, 1])
        point1.set_data([traj[i, 0]], [traj[i, 1]])
        step_text.set_text(f'步数: {i}\n损失: {losses[i]:.4f}')
        line2.set_data(range(i+1), losses[:i+1])
        return line1, point1, step_text, line2

    ani = animation.FuncAnimation(fig, animate, frames=len(traj),
                                  interval=200, blit=True, repeat=True)

    plt.tight_layout()
    try:
        ani.save('output_02d_梯度下降动画.gif', writer='pillow', fps=5)
        print("已保存: output_02d_梯度下降动画.gif")
    except Exception as e:
        print(f"保存GIF失败（{e}），直接显示动画")
    plt.show()


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Demo 02: 梯度下降")
    print("=" * 50)

    demo_learning_rate_effect()
    demo_ill_conditioned()
    demo_gradient_descent_animation()

    print("\n全部演示完成！")
    print("\n关键结论：")
    print("  1. 学习率太大→震荡发散，太小→收敛极慢")
    print("  2. 病态函数（各方向梯度差异大）上，普通梯度下降效率低")
    print("  3. 这就是为什么3DGS使用Adam优化器")

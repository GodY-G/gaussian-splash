"""
Demo 03: Adam优化器
===================
演示内容：
  1. 手写Adam优化器（不调库）
  2. Adam vs SGD 在病态函数上的对比
  3. Adam的自适应学习率机制可视化

运行：python demo_03_Adam优化器.py
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# 优化器实现
# ─────────────────────────────────────────────

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        return params - self.lr * grads

    def reset(self):
        pass


class SGDMomentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.momentum * self.v - self.lr * grads
        return params + self.v

    def reset(self):
        self.v = None


class Adam:
    """
    Adam优化器（手写实现）
    论文：Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)
    """
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1    # 一阶矩衰减系数（动量）
        self.beta2 = beta2    # 二阶矩衰减系数（自适应学习率）
        self.eps = eps        # 防止除零
        self.m = None         # 一阶矩（梯度的指数移动平均）
        self.v = None         # 二阶矩（梯度平方的指数移动平均）
        self.t = 0            # 时间步

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # 更新一阶矩（梯度的"动量"）
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads

        # 更新二阶矩（梯度平方的"动量"，用于自适应学习率）
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2

        # 偏差修正（早期步骤时m和v被初始化为0，需要修正）
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # 参数更新：学习率被 √v̂ 自适应缩放
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0


class RMSProp:
    def __init__(self, lr=0.01, decay=0.9, eps=1e-8):
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.decay * self.v + (1 - self.decay) * grads**2
        return params - self.lr * grads / (np.sqrt(self.v) + self.eps)

    def reset(self):
        self.v = None


# ─────────────────────────────────────────────
# 第一部分：各优化器在病态函数上的对比
# ─────────────────────────────────────────────

def run_optimizer(optimizer, f, grad_f, x0, n_steps=200):
    optimizer.reset()
    x = x0.copy().astype(float)
    trajectory = [x.copy()]
    losses = [f(*x)]
    for _ in range(n_steps):
        g = grad_f(*x)
        x = optimizer.step(x, g)
        trajectory.append(x.copy())
        losses.append(f(*x))
    return np.array(trajectory), np.array(losses)


def demo_optimizer_comparison():
    """在病态函数上对比各优化器"""
    # 病态函数：x方向和y方向梯度相差100倍
    f      = lambda x, y: 0.5 * (x**2 + 100*y**2)
    grad_f = lambda x, y: np.array([x, 100*y])

    x0 = np.array([1.5, 0.5])

    optimizers = [
        (SGD(lr=0.009),              'SGD (lr=0.009)',         'red'),
        (SGDMomentum(lr=0.009),      'SGD+Momentum (lr=0.009)', 'orange'),
        (RMSProp(lr=0.1),            'RMSProp (lr=0.1)',       'green'),
        (Adam(lr=0.1),               'Adam (lr=0.1)',          'blue'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("各优化器在病态函数上的对比\nf(x,y) = 0.5(x² + 100y²)",
                 fontsize=13, fontweight='bold')

    xf = np.linspace(-2, 2, 200)
    yf = np.linspace(-0.6, 0.6, 200)
    Xf, Yf = np.meshgrid(xf, yf)
    Zf = f(Xf, Yf)

    axes[0].contourf(Xf, Yf, Zf, levels=20, cmap='RdYlBu_r', alpha=0.5)
    axes[0].contour(Xf, Yf, Zf, levels=20, colors='gray', alpha=0.3, linewidths=0.5)
    axes[0].plot(0, 0, 'w*', markersize=15, zorder=6)
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-0.6, 0.6)
    axes[0].set_title('参数空间轨迹')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    axes[1].set_title('损失收敛曲线（对数坐标）')
    axes[1].set_xlabel('迭代步数')
    axes[1].set_ylabel('损失值')
    axes[1].grid(True, alpha=0.3)

    for opt, label, color in optimizers:
        traj, losses = run_optimizer(opt, f, grad_f, x0, n_steps=200)
        axes[0].plot(traj[:, 0], traj[:, 1], '-', color=color,
                     linewidth=1.5, alpha=0.8, label=label)
        axes[1].semilogy(np.clip(losses, 1e-10, None), color=color,
                         linewidth=2, label=f'{label} (最终: {losses[-1]:.2e})')

    axes[0].legend(fontsize=9, loc='upper right')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('output_03a_优化器对比.png', dpi=120, bbox_inches='tight')
    print("已保存: output_03a_优化器对比.png")
    plt.show()


# ─────────────────────────────────────────────
# 第二部分：Adam自适应学习率可视化
# ─────────────────────────────────────────────

def demo_adam_adaptive_lr():
    """可视化Adam的自适应学习率机制"""
    print("\n" + "="*60)
    print("Adam自适应学习率机制演示")
    print("="*60)

    # 模拟两个参数：一个梯度大，一个梯度小
    np.random.seed(42)
    n_steps = 100

    # 参数1：梯度大（对应y方向，梯度~100）
    grads_large = np.random.normal(50, 10, n_steps)
    # 参数2：梯度小（对应x方向，梯度~1）
    grads_small = np.random.normal(1, 0.2, n_steps)

    adam = Adam(lr=0.1)
    adam.reset()

    effective_lr_large = []
    effective_lr_small = []

    for g_large, g_small in zip(grads_large, grads_small):
        # 手动模拟Adam的有效学习率
        adam.t += 1
        if adam.m is None:
            adam.m = np.zeros(2)
            adam.v = np.zeros(2)

        g = np.array([g_large, g_small])
        adam.m = adam.beta1 * adam.m + (1 - adam.beta1) * g
        adam.v = adam.beta2 * adam.v + (1 - adam.beta2) * g**2

        m_hat = adam.m / (1 - adam.beta1**adam.t)
        v_hat = adam.v / (1 - adam.beta2**adam.t)

        # 有效学习率 = lr / √v̂
        eff_lr = adam.lr / (np.sqrt(v_hat) + adam.eps)
        effective_lr_large.append(eff_lr[0])
        effective_lr_small.append(eff_lr[1])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Adam自适应学习率机制\n梯度大的参数→有效学习率小；梯度小的参数→有效学习率大",
                 fontsize=12, fontweight='bold')

    steps = range(n_steps)

    # 梯度大小
    axes[0, 0].plot(steps, grads_large, 'r-', alpha=0.7, label='大梯度参数 (~50)')
    axes[0, 0].plot(steps, grads_small, 'b-', alpha=0.7, label='小梯度参数 (~1)')
    axes[0, 0].set_title('梯度大小')
    axes[0, 0].set_ylabel('梯度值')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 有效学习率
    axes[0, 1].plot(steps, effective_lr_large, 'r-', linewidth=2, label='大梯度参数的有效lr')
    axes[0, 1].plot(steps, effective_lr_small, 'b-', linewidth=2, label='小梯度参数的有效lr')
    axes[0, 1].set_title('Adam有效学习率（自适应！）')
    axes[0, 1].set_ylabel('有效学习率')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 对比：SGD的有效学习率是固定的
    axes[1, 0].axhline(y=0.1, color='r', linewidth=2, label='SGD：所有参数相同lr=0.1')
    axes[1, 0].plot(steps, effective_lr_large, 'r--', alpha=0.5, label='Adam大梯度参数')
    axes[1, 0].plot(steps, effective_lr_small, 'b--', alpha=0.5, label='Adam小梯度参数')
    axes[1, 0].set_title('SGD vs Adam 有效学习率对比')
    axes[1, 0].set_ylabel('有效学习率')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # 偏差修正的作用
    beta1, beta2 = 0.9, 0.999
    m, v = 0.0, 0.0
    m_no_correct, v_no_correct = [], []
    m_corrected, v_corrected = [], []
    g_const = 1.0  # 假设梯度恒为1

    for t in range(1, 51):
        m = beta1 * m + (1 - beta1) * g_const
        v = beta2 * v + (1 - beta2) * g_const**2
        m_no_correct.append(m)
        v_no_correct.append(v)
        m_corrected.append(m / (1 - beta1**t))
        v_corrected.append(v / (1 - beta2**t))

    axes[1, 1].plot(m_no_correct, 'r--', label='m（未修正）', linewidth=2)
    axes[1, 1].plot(m_corrected, 'r-', label='m̂（偏差修正后）', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='k', linestyle=':', label='真实梯度=1.0')
    axes[1, 1].set_title('偏差修正的作用\n（早期步骤m被低估，修正后更准确）')
    axes[1, 1].set_xlabel('步数')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('步数')

    plt.tight_layout()
    plt.savefig('output_03b_Adam自适应学习率.png', dpi=120, bbox_inches='tight')
    print("已保存: output_03b_Adam自适应学习率.png")
    plt.show()

    print("\nAdam关键结论：")
    print(f"  大梯度参数的有效学习率: ~{np.mean(effective_lr_large[-20:]):.4f}")
    print(f"  小梯度参数的有效学习率: ~{np.mean(effective_lr_small[-20:]):.4f}")
    print(f"  比值: {np.mean(effective_lr_small[-20:])/np.mean(effective_lr_large[-20:]):.1f}x")
    print("  → Adam自动为小梯度参数分配更大的学习率！")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Demo 03: Adam优化器")
    print("=" * 50)

    demo_optimizer_comparison()
    demo_adam_adaptive_lr()

    print("\n全部演示完成！")
    print("\n3DGS中Adam的作用：")
    print("  - 数百万个高斯基元，每个参数的梯度大小差异很大")
    print("  - Adam为每个参数自适应调整学习率")
    print("  - 位置参数、颜色参数、不透明度参数可以用不同的有效学习率更新")

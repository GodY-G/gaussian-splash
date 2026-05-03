"""
Demo 05: 反向传播可视化
=======================
演示内容：
  1. 手写计算图，逐节点反向传播
  2. 数值梯度 vs 解析梯度验证
  3. 3DGS梯度流可视化（简化版）
  4. 各参数梯度大小分布

与官方代码的对应：
  train.py: loss.backward() + optimizer.step()
  gaussian_renderer/__init__.py: render()

运行：python demo_05_反向传播可视化.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# 第一部分：手写计算图和反向传播
# ─────────────────────────────────────────────

class Node:
    """计算图节点"""
    def __init__(self, name, value=None):
        self.name = name
        self.value = value
        self.grad = 0.0
        self.inputs = []
        self.backward_fn = None

    def __repr__(self):
        return f"Node({self.name}={self.value:.4f}, grad={self.grad:.4f})"


def demo_computation_graph():
    """
    手动构建计算图并执行反向传播
    示例：f(x, y) = (x + y) * sin(x)
    """
    print("="*60)
    print("计算图反向传播演示")
    print("f(x, y) = (x + y) * sin(x)")
    print("="*60)

    # 前向传播
    x_val, y_val = 2.0, 3.0

    # 节点
    x = Node('x', x_val)
    y = Node('y', y_val)
    z1 = Node('z1=x+y', x_val + y_val)          # z1 = x + y
    z2 = Node('z2=sin(x)', np.sin(x_val))        # z2 = sin(x)
    out = Node('out=z1*z2', z1.value * z2.value) # out = z1 * z2

    print(f"\n前向传播：")
    print(f"  x = {x.value}")
    print(f"  y = {y.value}")
    print(f"  z1 = x + y = {z1.value}")
    print(f"  z2 = sin(x) = {z2.value:.4f}")
    print(f"  out = z1 * z2 = {out.value:.4f}")

    # 反向传播（手动链式法则）
    out.grad = 1.0  # 损失对输出的梯度（假设L=out）

    # ∂out/∂z1 = z2 = sin(x)
    z1.grad = out.grad * z2.value
    # ∂out/∂z2 = z1 = x+y
    z2.grad = out.grad * z1.value

    # ∂z1/∂x = 1, ∂z1/∂y = 1
    # ∂z2/∂x = cos(x)
    x.grad = z1.grad * 1.0 + z2.grad * np.cos(x_val)
    y.grad = z1.grad * 1.0

    print(f"\n反向传播（链式法则）：")
    print(f"  ∂L/∂out = {out.grad}")
    print(f"  ∂L/∂z1 = ∂L/∂out · ∂out/∂z1 = {out.grad} · sin({x_val:.1f}) = {z1.grad:.4f}")
    print(f"  ∂L/∂z2 = ∂L/∂out · ∂out/∂z2 = {out.grad} · {z1.value} = {z2.grad:.4f}")
    print(f"  ∂L/∂x  = ∂L/∂z1·1 + ∂L/∂z2·cos(x) = {z1.grad:.4f}·1 + {z2.grad:.4f}·{np.cos(x_val):.4f} = {x.grad:.4f}")
    print(f"  ∂L/∂y  = ∂L/∂z1·1 = {y.grad:.4f}")

    # 数值验证
    eps = 1e-5
    f = lambda x, y: (x + y) * np.sin(x)
    grad_x_numerical = (f(x_val+eps, y_val) - f(x_val-eps, y_val)) / (2*eps)
    grad_y_numerical = (f(x_val, y_val+eps) - f(x_val, y_val-eps)) / (2*eps)

    print(f"\n数值梯度验证（有限差分）：")
    print(f"  ∂L/∂x 解析 = {x.grad:.6f}，数值 = {grad_x_numerical:.6f}，误差 = {abs(x.grad-grad_x_numerical):.2e}")
    print(f"  ∂L/∂y 解析 = {y.grad:.6f}，数值 = {grad_y_numerical:.6f}，误差 = {abs(y.grad-grad_y_numerical):.2e}")

    # 可视化计算图
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title("计算图：f(x,y) = (x+y)·sin(x)\n蓝色=前向值，红色=反向梯度",
                 fontsize=12, fontweight='bold')

    # 节点位置
    positions = {
        'x':   (1, 6), 'y':   (1, 2),
        'z1':  (4, 4), 'z2':  (4, 6),
        'out': (7, 5),
    }
    node_data = {
        'x':   (x_val, x.grad),
        'y':   (y_val, y.grad),
        'z1':  (z1.value, z1.grad),
        'z2':  (z2.value, z2.grad),
        'out': (out.value, out.grad),
    }
    labels = {
        'x': 'x', 'y': 'y',
        'z1': 'z₁=x+y', 'z2': 'z₂=sin(x)',
        'out': 'out=z₁·z₂',
    }

    for name, (px, py) in positions.items():
        val, grad = node_data[name]
        circle = plt.Circle((px, py), 0.6, color='lightblue', zorder=3)
        ax.add_patch(circle)
        ax.text(px, py+0.1, labels[name], ha='center', va='center', fontsize=9, fontweight='bold', zorder=4)
        ax.text(px, py-0.25, f'={val:.3f}', ha='center', va='center', fontsize=8, color='blue', zorder=4)
        ax.text(px, py-0.55, f'grad={grad:.3f}', ha='center', va='center', fontsize=7, color='red', zorder=4)

    # 前向边
    edges_fwd = [('x','z1'), ('y','z1'), ('x','z2'), ('z1','out'), ('z2','out')]
    for src, dst in edges_fwd:
        sx, sy = positions[src]
        dx, dy = positions[dst]
        ax.annotate('', xy=(dx-0.6, dy), xytext=(sx+0.6, sy),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # 反向边（梯度流）
    edges_bwd = [('out','z1'), ('out','z2'), ('z1','x'), ('z1','y'), ('z2','x')]
    for src, dst in edges_bwd:
        sx, sy = positions[src]
        dx, dy = positions[dst]
        ax.annotate('', xy=(dx+0.6, dy+0.2), xytext=(sx-0.6, sy+0.2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5, linestyle='dashed'))

    legend_elements = [
        mpatches.Patch(color='blue', label='前向传播（计算值）'),
        mpatches.Patch(color='red', label='反向传播（梯度）'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig('output_05a_计算图.png', dpi=120, bbox_inches='tight')
    print("\n已保存: output_05a_计算图.png")
    plt.show()


# ─────────────────────────────────────────────
# 第二部分：数值梯度验证工具
# ─────────────────────────────────────────────

def numerical_gradient(f, params, eps=1e-5):
    """
    数值梯度（有限差分）
    用于验证解析梯度的正确性
    """
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        grad[i] = (f(params_plus) - f(params_minus)) / (2 * eps)
    return grad


def demo_gradient_check():
    """梯度检验：验证解析梯度的正确性"""
    print("\n" + "="*60)
    print("梯度检验（Gradient Check）")
    print("="*60)

    # 测试函数：简化的高斯渲染损失
    # f(μ, σ) = -log(N(target; μ, σ²))（负对数似然）
    target = np.array([2.0, 1.5])

    def gaussian_nll(params):
        """负对数似然损失"""
        mu = params[:2]
        log_sigma = params[2:4]
        sigma = np.exp(log_sigma)
        diff = target - mu
        return 0.5 * np.sum((diff/sigma)**2) + np.sum(log_sigma)

    def analytical_gradient(params):
        """解析梯度"""
        mu = params[:2]
        log_sigma = params[2:4]
        sigma = np.exp(log_sigma)
        diff = target - mu

        grad_mu = -(diff / sigma**2)
        grad_log_sigma = 1 - (diff**2 / sigma**2)
        return np.concatenate([grad_mu, grad_log_sigma])

    # 在随机点处检验
    np.random.seed(42)
    params = np.array([1.5, 1.0, 0.3, -0.2])

    grad_analytical = analytical_gradient(params)
    grad_numerical = numerical_gradient(gaussian_nll, params)

    print(f"\n参数: μ={params[:2]}, log_σ={params[2:]}")
    print(f"\n解析梯度: {np.round(grad_analytical, 6)}")
    print(f"数值梯度: {np.round(grad_numerical, 6)}")
    print(f"相对误差: {np.abs(grad_analytical - grad_numerical) / (np.abs(grad_numerical) + 1e-8)}")
    print(f"最大相对误差: {np.max(np.abs(grad_analytical - grad_numerical) / (np.abs(grad_numerical) + 1e-8)):.2e}")

    # 可视化：梯度检验的重要性
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("梯度检验：解析梯度 vs 数值梯度", fontsize=12, fontweight='bold')

    # 左图：两种梯度的对比
    ax = axes[0]
    x_pos = np.arange(len(grad_analytical))
    width = 0.35
    ax.bar(x_pos - width/2, grad_analytical, width, label='解析梯度', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, grad_numerical, width, label='数值梯度', color='red', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['∂/∂μ₁', '∂/∂μ₂', '∂/∂log_σ₁', '∂/∂log_σ₂'])
    ax.set_title('解析梯度 vs 数值梯度\n（两者应该几乎相同）')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 右图：不同eps下的数值梯度精度
    ax2 = axes[1]
    eps_values = np.logspace(-8, -1, 50)
    errors = []
    for eps in eps_values:
        grad_num = numerical_gradient(gaussian_nll, params, eps)
        err = np.max(np.abs(grad_analytical - grad_num))
        errors.append(err)

    ax2.loglog(eps_values, errors, 'b-', linewidth=2)
    ax2.set_title('数值梯度精度 vs eps\n（eps太小→数值误差，太大→截断误差）')
    ax2.set_xlabel('eps（有限差分步长）')
    ax2.set_ylabel('最大绝对误差')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(1e-5, color='red', linestyle='--', label='推荐eps=1e-5')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('output_05b_梯度检验.png', dpi=120, bbox_inches='tight')
    print("\n已保存: output_05b_梯度检验.png")
    plt.show()


# ─────────────────────────────────────────────
# 第三部分：3DGS梯度流可视化
# ─────────────────────────────────────────────

def demo_3dgs_gradient_flow():
    """
    模拟3DGS训练中各参数的梯度大小分布
    展示为什么不同参数需要不同的学习率
    """
    print("\n" + "="*60)
    print("3DGS训练梯度流分析")
    print("="*60)

    np.random.seed(42)
    n_gaussians = 1000
    n_steps = 100

    # 模拟各参数的梯度大小（基于3DGS论文的典型值）
    # 不同参数的梯度量级差异很大
    grad_scales = {
        'xyz（位置）':      0.0001,
        'rotation（旋转）': 0.001,
        'scaling（缩放）':  0.005,
        'opacity（不透明度）': 0.05,
        'sh_dc（颜色DC）':  0.01,
        'sh_rest（颜色高阶）': 0.0005,
    }

    # 官方代码中对应的学习率
    learning_rates = {
        'xyz（位置）':      0.00016,
        'rotation（旋转）': 0.001,
        'scaling（缩放）':  0.005,
        'opacity（不透明度）': 0.05,
        'sh_dc（颜色DC）':  0.0025,
        'sh_rest（颜色高阶）': 0.000125,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("3DGS训练梯度分析\n（理解为什么不同参数需要不同学习率）",
                 fontsize=12, fontweight='bold')

    # 梯度大小分布
    ax = axes[0, 0]
    param_names = list(grad_scales.keys())
    grad_means = [grad_scales[k] for k in param_names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(param_names)))
    bars = ax.bar(range(len(param_names)), grad_means, color=colors)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=30, ha='right', fontsize=8)
    ax.set_title('各参数梯度量级（典型值）')
    ax.set_ylabel('梯度均值')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # 学习率 vs 梯度大小
    ax2 = axes[0, 1]
    lr_values = [learning_rates[k] for k in param_names]
    ax2.scatter(grad_means, lr_values, c=colors, s=100, zorder=5)
    for i, name in enumerate(param_names):
        ax2.annotate(name.split('（')[0], (grad_means[i], lr_values[i]),
                     textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('梯度量级')
    ax2.set_ylabel('学习率')
    ax2.set_title('学习率 vs 梯度量级\n（梯度大→学习率大，梯度小→学习率小）')
    ax2.grid(True, alpha=0.3)

    # 模拟训练过程中的梯度变化
    ax3 = axes[1, 0]
    steps = np.arange(n_steps)
    for name, scale in list(grad_scales.items())[:3]:
        # 模拟梯度随训练步数的变化（逐渐减小）
        grads = scale * np.exp(-steps/50) * (1 + 0.3*np.random.randn(n_steps))
        grads = np.abs(grads)
        ax3.semilogy(steps, grads, linewidth=2, label=name)
    ax3.set_title('训练过程中梯度的变化\n（梯度逐渐减小→收敛）')
    ax3.set_xlabel('训练步数')
    ax3.set_ylabel('梯度均值（对数坐标）')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Adam有效学习率
    ax4 = axes[1, 1]
    beta2 = 0.999
    for name, scale in list(grad_scales.items())[:3]:
        lr = learning_rates[name]
        v = 0
        effective_lrs = []
        for step in range(1, n_steps+1):
            g = scale * (1 + 0.3*np.random.randn())
            v = beta2 * v + (1-beta2) * g**2
            v_hat = v / (1 - beta2**step)
            effective_lr = lr / (np.sqrt(v_hat) + 1e-8)
            effective_lrs.append(effective_lr)
        ax4.semilogy(steps, effective_lrs, linewidth=2, label=name)
    ax4.set_title('Adam有效学习率随训练步数的变化')
    ax4.set_xlabel('训练步数')
    ax4.set_ylabel('有效学习率（对数坐标）')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_05c_3DGS梯度流.png', dpi=120, bbox_inches='tight')
    print("已保存: output_05c_3DGS梯度流.png")
    plt.show()

    print("\n关键观察：")
    print("  - 不透明度梯度最大（0.05），学习率也最大（0.05）")
    print("  - 位置梯度最小（0.0001），学习率也最小（0.00016）")
    print("  - Adam自动适应这种差异，但手动设置不同学习率效果更好")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Demo 05: 反向传播可视化")
    print("=" * 60)

    demo_computation_graph()
    demo_gradient_check()
    demo_3dgs_gradient_flow()

    print("\n全部演示完成！")
    print("\n关键结论：")
    print("  1. 反向传播 = 链式法则在计算图上的系统应用")
    print("  2. 梯度检验（数值梯度）是验证实现正确性的重要工具")
    print("  3. 3DGS不同参数的梯度量级差异巨大，需要不同学习率")
    print("  4. 官方代码用PyTorch自动微分 + CUDA扩展计算梯度")

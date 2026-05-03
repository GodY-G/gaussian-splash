"""
Demo 04: 损失函数对比
=====================
演示内容：
  1. L1 vs L2 损失的特性对比
  2. SSIM（结构相似性）的计算原理
  3. 3DGS训练损失：L = (1-λ)L₁ + λL_SSIM
  4. 不同损失函数对图像重建的影响

运行：python demo_04_损失函数对比.py
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# 损失函数实现
# ─────────────────────────────────────────────

def l1_loss(pred, target):
    return np.mean(np.abs(pred - target))

def l2_loss(pred, target):
    return np.mean((pred - target)**2)

def ssim_loss(pred, target):
    """SSIM损失（1 - SSIM，越小越好）"""
    val = ssim_metric(pred, target, data_range=1.0)
    return 1 - val

def combined_loss(pred, target, lam=0.2):
    """3DGS训练损失：(1-λ)L₁ + λ(1-SSIM)"""
    return (1 - lam) * l1_loss(pred, target) + lam * ssim_loss(pred, target)


# ─────────────────────────────────────────────
# 第一部分：L1 vs L2 的特性
# ─────────────────────────────────────────────

def demo_l1_vs_l2_properties():
    """L1和L2损失的数学特性对比"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("L1 vs L2 损失函数特性", fontsize=13, fontweight='bold')

    # 左图：损失函数形状
    ax = axes[0]
    e = np.linspace(-3, 3, 300)
    ax.plot(e, np.abs(e),    'b-', linewidth=2.5, label='L1: |e|')
    ax.plot(e, e**2,          'r-', linewidth=2.5, label='L2: e²')
    ax.plot(e, np.abs(e)**1.5,'g--', linewidth=1.5, label='L1.5: |e|^1.5（参考）')
    ax.set_title('损失函数形状')
    ax.set_xlabel('误差 e = y - ŷ')
    ax.set_ylabel('损失值')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 9)

    # 中图：梯度（导数）
    ax = axes[1]
    ax.plot(e, np.sign(e),  'b-', linewidth=2.5, label='L1梯度: sign(e)（恒定±1）')
    ax.plot(e, 2*e,          'r-', linewidth=2.5, label='L2梯度: 2e（与误差成正比）')
    ax.set_title('梯度（导数）')
    ax.set_xlabel('误差 e')
    ax.set_ylabel('梯度值')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)

    # 右图：对异常值的敏感性
    ax = axes[2]
    errors_normal = np.array([0.1, -0.2, 0.15, -0.1, 0.05])
    errors_outlier = np.append(errors_normal, [2.0])  # 加一个异常值

    x_pos = np.arange(len(errors_outlier))
    ax.bar(x_pos[:-1], np.abs(errors_normal), color='lightblue', label='正常误差')
    ax.bar(x_pos[-1], np.abs(errors_outlier[-1]), color='red', alpha=0.7, label='异常值')

    l1_normal = np.mean(np.abs(errors_normal))
    l2_normal = np.mean(errors_normal**2)
    l1_outlier = np.mean(np.abs(errors_outlier))
    l2_outlier = np.mean(errors_outlier**2)

    ax.axhline(l1_normal, color='blue', linestyle='--', linewidth=2,
               label=f'L1均值（无异常）={l1_normal:.3f}')
    ax.axhline(l1_outlier, color='blue', linestyle='-', linewidth=2,
               label=f'L1均值（有异常）={l1_outlier:.3f}')
    ax.set_title(f'异常值的影响\nL1变化: {l1_normal:.3f}→{l1_outlier:.3f} (+{(l1_outlier/l1_normal-1)*100:.0f}%)\n'
                 f'L2变化: {l2_normal:.3f}→{l2_outlier:.3f} (+{(l2_outlier/l2_normal-1)*100:.0f}%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_04a_L1vsL2特性.png', dpi=120, bbox_inches='tight')
    print("已保存: output_04a_L1vsL2特性.png")
    plt.show()


# ─────────────────────────────────────────────
# 第二部分：SSIM的计算原理
# ─────────────────────────────────────────────

def compute_ssim_components(x, y, window_size=11):
    """手动计算SSIM的三个分量"""
    C1 = (0.01 * 1.0)**2
    C2 = (0.03 * 1.0)**2

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    luminance  = (2*mu_x*mu_y + C1) / (mu_x**2 + mu_y**2 + C1)
    contrast   = (2*sigma_x*sigma_y + C2) / (sigma_x**2 + sigma_y**2 + C2)
    structure  = (sigma_xy + C2/2) / (sigma_x*sigma_y + C2/2)

    return luminance, contrast, structure, luminance * contrast * structure


def demo_ssim_components():
    """展示SSIM的三个分量：亮度、对比度、结构"""
    np.random.seed(42)

    # 创建测试图像对
    size = 64
    original = np.zeros((size, size))
    # 画一个简单的建筑轮廓
    original[10:50, 10:50] = 0.8
    original[20:40, 20:40] = 0.3
    original[15:45, 30:35] = 0.9  # 窗户

    # 不同类型的失真
    distortions = {
        '亮度变化\n（L2敏感，SSIM不敏感）': original * 0.7,
        '高斯噪声\n（L1/L2和SSIM都敏感）': np.clip(original + np.random.normal(0, 0.1, original.shape), 0, 1),
        '模糊\n（SSIM敏感，L2不太敏感）': np.array([[np.mean(original[max(0,i-2):i+3, max(0,j-2):j+3])
                                                      for j in range(size)] for i in range(size)]),
        '结构破坏\n（SSIM最敏感）': np.roll(original, 5, axis=1),
    }

    fig, axes = plt.subplots(2, len(distortions)+1, figsize=(16, 8))
    fig.suptitle("SSIM vs L1/L2：不同失真类型的损失对比", fontsize=12, fontweight='bold')

    # 原始图像
    axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('原始图像', fontsize=10)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, '参考图像', ha='center', va='center',
                    fontsize=12, transform=axes[1, 0].transAxes)

    for idx, (name, distorted) in enumerate(distortions.items()):
        col = idx + 1
        axes[0, col].imshow(distorted, cmap='gray', vmin=0, vmax=1)
        axes[0, col].axis('off')

        l1 = l1_loss(distorted, original)
        l2 = l2_loss(distorted, original)
        ssim_val = ssim_metric(original, distorted, data_range=1.0)
        lum, cont, struct, ssim_manual = compute_ssim_components(
            original.ravel(), distorted.ravel())

        axes[0, col].set_title(name, fontsize=9)

        info = (f'L1 = {l1:.4f}\n'
                f'L2 = {l2:.4f}\n'
                f'SSIM = {ssim_val:.4f}\n'
                f'─────────\n'
                f'亮度: {lum:.3f}\n'
                f'对比度: {cont:.3f}\n'
                f'结构: {struct:.3f}')
        axes[1, col].text(0.05, 0.95, info, transform=axes[1, col].transAxes,
                          fontsize=9, va='top', family='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightyellow'))
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig('output_04b_SSIM分析.png', dpi=120, bbox_inches='tight')
    print("已保存: output_04b_SSIM分析.png")
    plt.show()


# ─────────────────────────────────────────────
# 第三部分：3DGS训练损失
# ─────────────────────────────────────────────

def demo_3dgs_loss():
    """演示3DGS训练损失 L = (1-λ)L₁ + λ(1-SSIM)"""
    print("\n" + "="*60)
    print("3DGS训练损失：L = (1-λ)L₁ + λ·L_D-SSIM，λ=0.2")
    print("="*60)

    np.random.seed(0)
    size = 64
    target = np.random.rand(size, size) * 0.5 + 0.25

    # 模拟不同质量的"渲染结果"
    qualities = np.linspace(0, 1, 20)
    l1_vals, l2_vals, ssim_vals, combined_vals = [], [], [], []

    for q in qualities:
        pred = q * target + (1 - q) * np.random.rand(size, size)
        l1_vals.append(l1_loss(pred, target))
        l2_vals.append(l2_loss(pred, target))
        ssim_vals.append(ssim_loss(pred, target))
        combined_vals.append(combined_loss(pred, target, lam=0.2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("3DGS训练损失随渲染质量的变化", fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(qualities, l1_vals,       'b-o', markersize=4, linewidth=2, label='L1损失')
    ax.plot(qualities, ssim_vals,     'r-s', markersize=4, linewidth=2, label='1-SSIM损失')
    ax.plot(qualities, combined_vals, 'g-^', markersize=4, linewidth=2,
            label='3DGS损失 (1-0.2)L₁ + 0.2·SSIM', linewidth=2.5)
    ax.set_xlabel('渲染质量（0=随机噪声，1=完美重建）')
    ax.set_ylabel('损失值')
    ax.set_title('各损失函数随质量的变化')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # 不同λ值的影响
    ax2 = axes[1]
    lambdas = [0.0, 0.1, 0.2, 0.5, 1.0]
    for lam in lambdas:
        vals = [(1-lam)*l1 + lam*s for l1, s in zip(l1_vals, ssim_vals)]
        ax2.plot(qualities, vals, linewidth=2,
                 label=f'λ={lam}' + (' ← 3DGS默认' if lam == 0.2 else ''))
    ax2.set_xlabel('渲染质量')
    ax2.set_ylabel('损失值')
    ax2.set_title('不同λ值的影响\nL = (1-λ)L₁ + λ·(1-SSIM)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig('output_04c_3DGS损失.png', dpi=120, bbox_inches='tight')
    print("已保存: output_04c_3DGS损失.png")
    plt.show()


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Demo 04: 损失函数对比")
    print("=" * 50)

    demo_l1_vs_l2_properties()
    demo_ssim_components()
    demo_3dgs_loss()

    print("\n全部演示完成！")
    print("\n关键结论：")
    print("  1. L1对异常值不敏感，梯度恒定（训练稳定）")
    print("  2. L2对大误差惩罚更重，但对异常值敏感")
    print("  3. SSIM衡量结构相似性，更符合人眼感知")
    print("  4. 3DGS用 0.8·L1 + 0.2·SSIM，兼顾稳定性和感知质量")

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MyAdam:
    """手写 Adam 优化器"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1          # 一阶矩衰减率
        self.beta2 = beta2          # 二阶矩衰减率
        self.eps = eps              # 防止除零
        self.t = 0                  # 时间步计数
        self.m = None               # 一阶矩（动量）
        self.v = None               # 二阶矩（RMS）

    def step(self, param, grad):
        """
        Adam 更新公式：
          m_t = β₁·m_{t-1} + (1-β₁)·g_t        (一阶矩估计)
          v_t = β₂·v_{t-1} + (1-β₂)·g_t²       (二阶矩估计)
          m̂_t = m_t / (1-β₁ᵗ)                    (偏差修正)
          v̂_t = v_t / (1-β₂ᵗ)                    (偏差修正)
          θ_t = θ_{t-1} - lr·m̂_t / (√v̂_t + ε)   (参数更新)
        """
        self.t += 1

        # 初始化（第一次调用时）
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)

        # 更新有偏一阶和二阶矩估计
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        # 偏差修正
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # 参数更新
        param = param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return param


# ============ 测试：f(θ) = (θ - 5)² 最小值 θ*=5 ============

def f(theta):
    return (theta - 5)**2

def grad_f(theta):
    return 2 * (theta - 5)


# --- Adam ---
theta_adam = 0.0
adam = MyAdam(lr=0.1, beta1=0.9, beta2=0.999)
history_adam = [theta_adam]

for _ in range(100):
    g = grad_f(theta_adam)
    theta_adam = adam.step(theta_adam, g)
    history_adam.append(theta_adam)

# --- SGD ---
theta_sgd = 0.0
lr_sgd = 0.1
history_sgd = [theta_sgd]

for _ in range(100):
    g = grad_f(theta_sgd)
    theta_sgd = theta_sgd - lr_sgd * g
    history_sgd.append(theta_sgd)

# --- SGD + Momentum ---
theta_mom = 0.0
velocity = 0.0
momentum = 0.9
lr_mom = 0.1
history_mom = [theta_mom]

for _ in range(100):
    g = grad_f(theta_mom)
    velocity = momentum * velocity - lr_mom * g
    theta_mom = theta_mom + velocity
    history_mom.append(theta_mom)

print(f"Adam最终: {history_adam[-1]:.6f} (应接近5.0)")
print(f"SGD最终:  {history_sgd[-1]:.6f}")
print(f"SGD+Momentum最终: {history_mom[-1]:.6f}")

# 画图
plt.figure(figsize=(10, 5))
plt.plot(history_adam, 'b-', linewidth=2, label='Adam (lr=0.1)')
plt.plot(history_sgd, 'r--', linewidth=1.5, label='SGD (lr=0.1)')
plt.plot(history_mom, 'g-.', linewidth=1.5, label='SGD+Momentum (lr=0.1)')
plt.axhline(5.0, color='k', linestyle=':', label='最优解 θ=5')
plt.xlabel('迭代步数')
plt.ylabel('θ 值')
plt.title('Adam vs SGD vs SGD+Momentum  求解 min (θ-5)²')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('B4_Adam_vs_SGD.png', dpi=120, bbox_inches='tight')
plt.show()
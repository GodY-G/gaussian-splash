class MyAdam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = 0.0
        self.v = 0.0
        self.t = 0

    def step(self, param, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return param - self.lr * m_hat / (v_hat**0.5 + self.eps)

# 测试：找到 f(θ)=(θ-5)² 的最小值
adam = MyAdam(lr=0.1)
theta = 0.0
for _ in range(100):
    grad = 2 * (theta - 5)     # ∇f = 2(θ-5)
    theta = adam.step(theta, grad)
print(f"Adam 结果: {theta:.3f}")   # 应接近 5.0
# B4. Adam 手写实现 —— 逐行详解

> 对应练习题：第 3-4 周 B4（手写 Adam 优化器）

---

## 零、Python 基础速览

```python
# 1. 变量：不用声明类型，直接赋值
x = 5           # 整数
y = 3.14        # 浮点数
name = "Adam"   # 字符串

# 2. 函数：def 定义，缩进表示函数体
def add(a, b):
    return a + b

# 3. 类：class 定义，__init__ 是构造函数
class Dog:
    def __init__(self, name):    # self = 对象本身
        self.name = name         # self.name = 这个对象的name属性

# 4. 列表 vs 数组
a = [1, 2, 3]           # Python原生列表
b = np.array([1, 2, 3]) # numpy数组，可以做数学运算
```

### 为什么每个方法参数都有 `self`？

```python
class MyAdam:
    def __init__(self, lr=0.001):   # self 出现在每个方法的第一参数
        self.lr = lr

    def step(self, param, grad):    # self 也必须在这里
        ...
```

| 概念 | 解释 |
|------|------|
| **类 (class)** | 一张"蓝图"，描述了对象长什么样、能做什么 |
| **对象 (object)** | 根据蓝图造出来的"实体" |
| **self** | 代表**当前这个对象本身** |

```python
# 创建两个不同的 Adam 对象
adam1 = MyAdam(lr=0.1)    # adam1 是对象1
adam2 = MyAdam(lr=0.01)   # adam2 是对象2

# 当调用 adam1.step(param, grad) 时：
# Python 自动把 adam1 传给 self
# → self 就是 adam1，所以 self.lr = 0.1

# 当调用 adam2.step(param, grad) 时：
# Python 自动把 adam2 传给 self
# → self 就是 adam2，所以 self.lr = 0.01
```

**一句话总结**：`self` 让每个对象知道"自己的数据是什么"。没有 `self`，`adam1` 和 `adam2` 就无法区分各自的 `lr`、`m`、`v`。

**调用时不需要传 self**：`adam.step(param, grad)` 只传两个参数，Python 自动把 `adam` 填到 `self` 的位置。

---

### `__init__` 中的 `lr`、`beta1`、`beta2` 是干什么的？

```python
def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
```

| 参数 | 全称 | 默认值 | 作用 |
|------|------|:------:|------|
| **lr** | Learning Rate（学习率） | 0.001 | 控制每次更新参数的**步长**。lr 太大→震荡/发散；lr 太小→收敛慢 |
| **beta1** | 一阶矩衰减率 | 0.9 | 控制梯度**方向**的"记忆长度"。越大惯性越强，旧梯度影响越久 |
| **beta2** | 二阶矩衰减率 | 0.999 | 控制梯度**大小**的"记忆长度"。越大对历史梯度大小越依赖 |
| **eps** | epsilon（防止除零） | 1e-8 | 极小的常数，防止 `√v̂` 为 0 时除零报错 |

**beta1 的直观理解**：

```
beta1 = 0.9  → 旧梯度权重 90%，新梯度 10% → 惯性大，方向平滑
beta1 = 0.5  → 新旧各一半 → 方向变化剧烈
beta1 = 0.0  → 退化为 SGD（完全没有动量记忆）
```

**beta2 的直观理解**：

```
beta2 = 0.999 → 二阶矩变化极其缓慢 → 自适应学习率很稳定
beta2 = 0.9   → 二阶矩快速响应新梯度 → 自适应学习率波动大
beta2 = 0.0   → 二阶矩完全失去平滑效果
```

**在 Adam 公式中的位置**：

```
一阶矩：m = beta1 × m_旧 + (1-beta1) × g_当前
二阶矩：v = beta2 × v_旧 + (1-beta2) × g_当前²
参数更新：θ = θ - lr × m̂ / (√v̂ + eps)
```

---

## 一、导入库

```python
import numpy as np
import matplotlib.pyplot as plt
```

| 语句 | 含义 |
|------|------|
| `import numpy as np` | 导入 numpy（数学计算库），起别名 `np`。之后写 `np.xxx` 调用 |
| `import matplotlib.pyplot as plt` | 导入 matplotlib 的绘图子模块，起别名 `plt` |

### numpy 是什么？为什么不用 Python 原生 list？

```python
a = [1, 2, 3]             # Python 原生 list
b = np.array([1, 2, 3])   # numpy ndarray

a * 2   # → [1, 2, 3, 1, 2, 3]    list 的 * 是"复制拼接"
b * 2   # → array([2, 4, 6])       ndarray 的 * 是"逐元素数学乘法"

a + 1   # → 报错！list 不能和整数相加
b + 1   # → array([2, 3, 4])       每个元素都 +1
```

**numpy 的核心优势**：整个数组作为一个整体做数学运算，不需要写 for 循环。

```python
# 中文字体设置，防止画图时中文变成方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# 防止负号显示为方块
plt.rcParams['axes.unicode_minus'] = False
```

---

## 二、`MyAdam` 类 —— 逐行讲解

### 完整代码骨架

```python
class MyAdam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = None
        self.v = None

    def step(self, param, grad):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        param = param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param
```

---

### 2.1 `__init__` 构造函数 —— 逐行讲解

```python
def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
```

| 语法元素 | 含义 |
|---------|------|
| `__init__` | 构造函数（前后各两个下划线）。创建对象时**自动调用** |
| `self` | 对象本身 |
| `lr=0.001` | 默认参数：调用时不传 `lr` 就用 0.001 |

```python
adam1 = MyAdam()             # 全部用默认值
adam2 = MyAdam(lr=0.1)       # lr 覆盖为 0.1，其余用默认值
adam3 = MyAdam(lr=0.01, beta1=0.95)  # 覆盖 lr 和 beta1
```

---

```python
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
```

**`self.xxx = xxx` 的含义**：把参数值"绑定"到对象上，以后对象的所有方法都能通过 `self.xxx` 访问。

```python
adam = MyAdam(lr=0.1)
print(adam.lr)      # → 0.1  （通过 . 访问对象的属性）
print(adam.beta1)   # → 0.9
```

---

```python
        self.t = 0
```

**时间步计数器**，记录 `step()` 被调用了多少次。用于偏差修正公式中的 `β₁ᵗ` 和 `β₂ᵗ`。

```
t=1 时：β₁ᵗ = 0.9¹  = 0.9
t=2 时：β₁ᵗ = 0.9²  = 0.81
t=10 时：β₁ᵗ = 0.9¹⁰ ≈ 0.35
t=100 时：β₁ᵗ ≈ 0.000027（接近 0，偏差修正几乎不起作用）
```

---

```python
        self.m = None
        self.v = None
```

初始化为 `None` 而不是 0。原因：第一次调用 `step()` 时，我们**不知道参数是什么形状**（标量？向量？矩阵？），等拿到梯度 `grad` 之后，再用 `np.zeros_like(grad)` 创建匹配形状的全零数组。

```python
# 如果 param 是标量 → grad 是标量 → m 创建为标量 0
# 如果 param 是 3×3 矩阵 → grad 是 3×3 矩阵 → m 创建为 3×3 零矩阵
```

---

### 2.2 `step` 方法 —— 逐行讲解

```python
    def step(self, param, grad):
```

| 参数 | 含义 |
|------|------|
| `param` | 当前参数值，需要被更新（可以是标量、向量、矩阵） |
| `grad` | 损失函数对 param 的梯度，形状与 param 相同 |

---

```python
        self.t += 1
```

Python 中 `x += 1` 等价于 `x = x + 1`。每次调用 step，时间计数 +1。

---

```python
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
```

**逐词拆解**：

| 写法 | 含义 |
|------|------|
| `self.m is None` | 判断 m 是不是从未被初始化（`is` 比 `==` 更适合判断 None） |
| `np.zeros_like(grad)` | 创建一个**和 grad 形状完全相同、值全为 0** 的 numpy 数组 |

```python
# 举例 1：grad 是标量 2.0
np.zeros_like(2.0)    # → array(0.)  （0维数组，等同于标量 0）

# 举例 2：grad 是一维数组 [0.5, -0.3, 0.1]
np.zeros_like([0.5, -0.3, 0.1])  # → array([0., 0., 0.])

# 举例 3：grad 是 2×2 矩阵
np.zeros_like([[1,2],[3,4]])  # → array([[0., 0.], [0., 0.]])
```

**为什么初始化为全零？** Adam 需要"累计"梯度信息。就像水桶从空开始接水——m 和 v 从零开始，每步累加新的梯度信息。

---

```python
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
```

**Adam 一阶矩更新公式**：`m = β₁·m_旧 + (1-β₁)·g_当前`

这行是做**指数加权移动平均**（Exponential Moving Average, EMA）：

```python
# 逐步演算（beta1=0.9）：
# step 1: m = 0.9×0.0 + 0.1×g₁ = 0.1·g₁
# step 2: m = 0.9×0.1·g₁ + 0.1×g₂ = 0.09·g₁ + 0.1·g₂
# step 3: m = 0.081·g₁ + 0.09·g₂ + 0.1·g₃
# ...
# 越早的梯度，权重越小（指数衰减）
```

一阶矩 m 的本质：**梯度的平滑方向**。相当于给梯度加了"惯性"。

---

```python
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
```

**Adam 二阶矩更新公式**：`v = β₂·v_旧 + (1-β₂)·g²`

| 符号 | 含义 |
|------|------|
| `grad**2` | 梯度的**平方**。Python 中 `**` 是幂运算符，`x**2` = x² |
| `beta2=0.999` | 二阶矩对历史非常依赖（99.9% 来自旧值，0.1% 来自新值） |

二阶矩 v 的本质：**梯度大小的平滑估计**。v 大 → 这个参数梯度一直很大 → 学习率应该小一点；v 小 → 梯度一直很小 → 学习率应该大一点。

```python
# 举例：grad = 2.0, v_旧 = 1.0, beta2 = 0.999
# grad**2 = 4.0
# v_新 = 0.999 × 1.0 + 0.001 × 4.0 = 0.999 + 0.004 = 1.003
```

---

```python
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
```

**偏差修正**。`self.beta1**self.t` 表示 `β₁ᵗ`（β₁ 的 t 次方）。

### 为什么需要修正？

因为 m 和 v 初始化为 0，前几步的估计**严重偏小**：

| 时刻 | m 的期望值 | 实际值（因为从 0 初始化） | 偏差 |
|------|-----------|-------------------------|------|
| t=1 | g₁ | 0.1·g₁ | 小了 10 倍 |
| t=2 | ≈g₁ | 0.19·g₁ | 小了 ~5 倍 |
| t=10 | ≈g₁ | ≈g₁ | 几乎无偏差 |

除以 `(1-β₁ᵗ)` 消除这个"冷启动"偏差：

```python
# t=1 时：
# β₁ᵗ = 0.9¹ = 0.9
# 1 - 0.9 = 0.1
# m_hat = m / 0.1 = m × 10   ← 补偿了 10 倍的初始偏差

# t=100 时：
# β₁ᵗ = 0.9¹⁰⁰ ≈ 0.000027
# 1 - β₁ᵗ ≈ 1.0
# m_hat ≈ m   ← 几乎不需要修正了
```

---

```python
        param = param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

**Adam 的参数更新公式**。这是整个优化器的核心。

| 部分 | 作用 |
|------|------|
| `self.lr` | 基础学习率，控制全局更新步长 |
| `m_hat` | 修正后的"梯度方向"（朝哪个方向走） |
| `np.sqrt(v_hat)` | 修正后的"梯度大小"的开方（步长缩放到合理范围） |
| `self.eps = 1e-8` | 防止分母为 0 的极小常数（0.00000001） |

### 为什么除以 √v̂？

这是 Adam 最精妙的设计——**自适应学习率**：

```
参数 A 的梯度一直很大（v̂_A = 100）
  → √v̂_A = 10
  → 有效学习率 = lr / 10  ← 自动变小，避免震荡

参数 B 的梯度一直很小（v̂_B = 0.01）
  → √v̂_B = 0.1
  → 有效学习率 = lr / 0.1  ← 自动变大，加速收敛
```

这就是为什么 Adam 能同时处理好"梯度大的参数"和"梯度小的参数"。

```python
        return param
```

返回更新后的参数值。注意：Adam 不存储参数——参数由调用方维护，step 只负责根据梯度算出新值并返回。

---

## 三、测试部分 —— 逐行讲解

### 3.1 创建测试函数

```python
def f(theta):
    return (theta - 5)**2

def grad_f(theta):
    return 2 * (theta - 5)
```

| 函数 | 公式 | 最小值 |
|------|------|:------:|
| `f(theta)` | (θ - 5)² | θ=5 时，值为 0 |
| `grad_f(theta)` | 2(θ - 5) | f 对 θ 求导的结果 |

`(theta - 5)**2`：** 是幂运算符。`2**3` = 2³ = 8。

### 3.2 运行 Adam

```python
theta_adam = 0.0                     # ① 起始点 θ = 0（离最优解 5 有距离）
adam = MyAdam(lr=0.1)                # ② 创建 MyAdam 对象，学习率设为 0.1
history_adam = [theta_adam]          # ③ 记录历史轨迹，初始值放入列表 [0.0]
```

```python
for _ in range(100):                 # ④ 循环 100 次
    g = grad_f(theta_adam)           # ⑤ 计算当前梯度
    theta_adam = adam.step(theta_adam, g)  # ⑥ 用 Adam 更新参数
    history_adam.append(theta_adam)  # ⑦ 记录更新后的 θ
```

**逐行解读**：

| 行 | 语法 | 解释 |
|----|------|------|
| ④ | `for _ in range(100):` | `_` 是 Python 惯例——"不关心循环变量值，只想重复 100 次" |
| ④ | `range(100)` | 生成 0, 1, 2, ..., 99 的序列，共 100 个数 |
| ⑤ | `grad_f(theta_adam)` | 调用梯度函数，算出当前梯度 = 2(θ-5) |
| ⑥ | `adam.step(...)` | 传入参数和梯度，Adam 内部用 m、v、偏差修正算出新参数 |
| ⑦ | `.append(...)` | 列表的方法，往末尾添加元素 |

```python
# .append() 示例：
a = [0]
a.append(0.5)      # a → [0, 0.5]
a.append(1.0)      # a → [0, 0.5, 1.0]
```

### 3.3 运行 SGD（对比）

```python
theta_sgd = 0.0
lr_sgd = 0.1
history_sgd = [theta_sgd]

for _ in range(100):
    g = grad_f(theta_sgd)
    theta_sgd = theta_sgd - lr_sgd * g    # SGD：θ = θ - lr × 梯度
    history_sgd.append(theta_sgd)
```

SGD 只有一个操作：**参数减去学习率 × 梯度**。没有动量（m），没有自适应学习率（v）。

### 3.4 运行 SGD+Momentum（额外对比）

```python
theta_mom = 0.0
velocity = 0.0             # 动量变量，初始为 0
momentum = 0.9
lr_mom = 0.1
history_mom = [theta_mom]

for _ in range(100):
    g = grad_f(theta_mom)
    velocity = momentum * velocity - lr_mom * g   # 动量更新
    theta_mom = theta_mom + velocity              # 参数更新
    history_mom.append(theta_mom)
```

`velocity` 类似于 Adam 的 `m`（一阶矩），但没有归一化和偏差修正。

### 3.5 打印结果

```python
print(f"Adam最终: {history_adam[-1]:.6f} (应接近5.0)")
print(f"SGD最终:  {history_sgd[-1]:.6f}")
print(f"SGD+Momentum最终: {history_mom[-1]:.6f}")
```

**f-string 语法**：

```python
name = "Adam"
x = 3.1415926
print(f"{name}的值是{x:.2f}")   # → "Adam的值是3.14"

# {: .2f}  → 保留 2 位小数的浮点数
# {: .6f}  → 保留 6 位小数的浮点数
```

**负索引语法**：

```python
a = [10, 20, 30, 40]
a[0]   # → 10    （第一个）
a[1]   # → 20    （第二个）
a[-1]  # → 40    （最后一个，倒数第一）
a[-2]  # → 30    （倒数第二）
```

---

## 四、绘图部分 —— 逐行讲解

```python
plt.figure(figsize=(10, 5))
```

创建一张画布。`figsize=(10,5)` 表示宽 10 英寸、高 5 英寸。

```python
plt.plot(history_adam, 'b-', linewidth=2, label='Adam (lr=0.1)')
```

| 参数 | 含义 |
|------|------|
| `history_adam` | y 轴数据（x 轴自动为 0, 1, 2, ..., 100） |
| `'b-'` | 蓝色实线。b=blue, -=实线, --=虚线, -.=点划线, :=点线 |
| `linewidth=2` | 线宽 2 像素 |
| `label='...'` | 图例标签，`plt.legend()` 会显示它 |

```python
plt.plot(history_sgd, 'r--', linewidth=1.5, label='SGD (lr=0.1)')
plt.plot(history_mom, 'g-.', linewidth=1.5, label='SGD+Momentum (lr=0.1)')
```

三根线：Adam 蓝色实线、SGD 红色虚线、SGD+Momentum 绿色点划线。

```python
plt.axhline(5.0, color='k', linestyle=':', label='最优解 θ=5')
```

| 函数 | 含义 |
|------|------|
| `axhline(y)` | 画一条**水平线**（Axis Horizontal Line），位置在 y=5.0 |
| `color='k'` | 黑色（k=black，b 已经被 blue 占用了） |
| `linestyle=':'` | 点线 |

```python
plt.xlabel('迭代步数')      # x 轴标签
plt.ylabel('θ 值')          # y 轴标签
plt.title('Adam vs SGD vs SGD+Momentum  求解 min (θ-5)²')  # 图标题
plt.legend()                # 显示图例（根据各 plot 的 label 参数自动生成）
plt.grid(True, alpha=0.3)   # 显示网格。alpha=不透明度（0=完全透明, 1=完全不透明）
```

```python
plt.savefig('B4_Adam_vs_SGD.png', dpi=120, bbox_inches='tight')
```

| 参数 | 含义 |
|------|------|
| `'B4_Adam_vs_SGD.png'` | 保存的文件名 |
| `dpi=120` | 分辨率 120 dots per inch |
| `bbox_inches='tight'` | 自动裁剪多余白边 |

```python
plt.show()
```

弹出窗口显示图形。

---

## 五、三个优化器对比总结

| 优化器 | 核心公式 | 特点 | 收敛速度 |
|--------|---------|------|:------:|
| **SGD** | `θ = θ - lr·g` | 最简单，但容易在峡谷中振荡 | 慢 |
| **SGD+Momentum** | `v=β·v-lr·g; θ=θ+v` | 增加惯性，缓解振荡 | 中 |
| **Adam** | `θ = θ - lr·m̂/(√v̂+ε)` | 动量 + 自适应学习率 + 偏差修正 | **快** |

### 为什么 Adam 在这个问题上比 SGD 快？

函数 `f(θ) = (θ-5)²` 的特点是：离最优解越远，梯度越大。

```
θ=0 时：梯度 = -10  （很大，SGD 会冲过头 → 震荡）
θ=4 时：梯度 = -2   （正常）
θ=5 时：梯度 = 0    （最优）
```

- **SGD**：用固定学习率 0.1，θ=0 时 `-10×0.1 = -1` 步长很大，但越靠近 5 步长不变 → 绕圈震荡
- **Adam**：二阶矩 `v` 记录了大梯度 → `√v̂` 变大 → 有效学习率自动缩小 → 大梯度时不冲过头，小梯度时不磨洋工

---

## 六、Adam 公式速查表

| 步骤 | 公式 | 代码 |
|------|------|------|
| 时间步 | t = t + 1 | `self.t += 1` |
| 一阶矩 | m = β₁m + (1-β₁)g | `self.m = self.beta1*self.m + (1-self.beta1)*grad` |
| 二阶矩 | v = β₂v + (1-β₂)g² | `self.v = self.beta2*self.v + (1-self.beta2)*grad**2` |
| 偏差修正 m̂ | m̂ = m/(1-β₁ᵗ) | `m_hat = self.m/(1-self.beta1**self.t)` |
| 偏差修正 v̂ | v̂ = v/(1-β₂ᵗ) | `v_hat = self.v/(1-self.beta2**self.t)` |
| 参数更新 | θ = θ - lr·m̂/(√v̂+ε) | `param - self.lr*m_hat/(np.sqrt(v_hat)+self.eps)` |

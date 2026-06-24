# 3D Gaussian Splatting 核心原理笔记

> 本文档对应论文：*"3D Gaussian Splatting for Real-Time Radiance Field Rendering"*, Kerbl et al., SIGGRAPH 2023
> 代码参考：diff-gaussian-rasterization CUDA 实现
> 生成日期：2026-06-24

---

# 1. 3D 高斯投影推导

## 1.1 3D 高斯的数学表示

一个三维高斯体由其均值 $\boldsymbol{\mu}$（3维向量）和协方差矩阵 $\boldsymbol{\Sigma}$（$3\times 3$ 对称正定矩阵）完全定义：

$$
G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}
$$

**物理意义**：$\mathbf{x} \in \mathbb{R}^3$ 是空间中的任意点坐标，$G(\mathbf{x})$ 表示该点处高斯体的"密度"值（未归一化，最大值为1）。由于我们使用高斯体作为场景表示的基本基元，每个高斯体代表场景中的一团辐射能量，其空间分布由 $\boldsymbol{\mu}$（位置）和 $\boldsymbol{\Sigma}$（形状、朝向）刻画。

| 数学符号 | 物理意义 | 代码变量 |
|---|---|---|
| $\boldsymbol{\mu}$ | 高斯均值，表示3D空间中的位置 | means3D / orig_points (float3, 3*P) |
| $\boldsymbol{\Sigma}$ | 3D协方差矩阵，控制形状和朝向 | cov3D (float[6], 对称矩阵上三角) |

## 1.2 协方差矩阵的分解参数化

直接优化 $\boldsymbol{\Sigma}$ 不可行，因为协方差矩阵必须保持半正定。3DGS 将 $\boldsymbol{\Sigma}$ 分解为**旋转矩阵 $\mathbf{R}$** 和**缩放矩阵 $\mathbf{S}$**：

$$
\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^\top \mathbf{R}^\top
$$

其中：

- $\mathbf{S} = \text{diag}(s_x, s_y, s_z)$ 是缩放对角矩阵
- $\mathbf{R}$ 是由**单位四元数** $\mathbf{q} = (q_w, q_x, q_y, q_z)$ 转化得到的旋转矩阵

**四元数转旋转矩阵公式**：

$$
\mathbf{R} = \begin{pmatrix}
1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1 - 2(q_x^2 + q_y^2)
\end{pmatrix}
$$

## 1.3 3D 到 2D 的投影（EWA Splatting）

将 3D 高斯体投影到 2D 图像平面需要两步：**视角变换 (View Transform)** 和 **仿射投影近似 (Jacobian Approximation)**。

### 1.3.1 视角变换

首先将世界坐标系下的高斯均值 $\boldsymbol{\mu}_w$ 变换到相机坐标系：

$$
\boldsymbol{\mu}_v = \mathbf{W} \cdot \boldsymbol{\mu}_w
$$

其中 $\mathbf{W} \in \mathbb{R}^{4\times 4}$ 是 world-to-view 变换矩阵（即视图矩阵）。

在代码中，这一步由 `transformPoint4x3` 完成：

```cuda
float3 t = transformPoint4x3(mean, viewmatrix);
// t.x = matrix[0]*p.x + matrix[4]*p.y + matrix[8]*p.z + matrix[12]
// t.y = matrix[1]*p.x + matrix[5]*p.y + matrix[9]*p.z + matrix[13]
// t.z = matrix[2]*p.x + matrix[6]*p.y + matrix[10]*p.z + matrix[14]
```

### 1.3.2 2D 协方差公式

根据 Zwicker et al. (EWA Splatting, 2002) 的结论，3D 高斯投影到 2D 图像平面后的协方差矩阵为：

$$
\boldsymbol{\Sigma}' = \mathbf{J} \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^\top \mathbf{J}^\top
$$

其中：

- $\mathbf{W}$ 是 world-to-view 变换矩阵的**左上 $3\times 3$ 子矩阵**（即只处理旋转部分，忽略平移）
- $\mathbf{J}$ 是投影变换的**仿射近似的雅可比矩阵**

### 1.3.3 雅可比矩阵推导

针孔相机模型下，从相机坐标系 $(x, y, z)$ 到 NDC 坐标系 $(x', y')$ 的投影为：

$$
x' = \frac{f_x \cdot x}{z}, \quad y' = \frac{f_y \cdot y}{z}
$$

其中 $f_x, f_y$ 是焦距（像素单位）。该投影映射的雅可比矩阵为：

$$
\mathbf{J} = \begin{pmatrix}
\frac{\partial x'}{\partial x} & \frac{\partial x'}{\partial y} & \frac{\partial x'}{\partial z} \\
\frac{\partial y'}{\partial x} & \frac{\partial y'}{\partial y} & \frac{\partial y'}{\partial z} \\
0 & 0 & 0
\end{pmatrix}
= \begin{pmatrix}
\frac{f_x}{z} & 0 & -\frac{f_x \cdot x}{z^2} \\
0 & \frac{f_y}{z} & -\frac{f_y \cdot y}{z^2} \\
0 & 0 & 0
\end{pmatrix}
$$

**代码实现**（`forward.cu` 的 `computeCov2D`）：

```cuda
glm::mat3 J = glm::mat3(
    focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
    0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
    0, 0, 0);
```

注意第三行全零——这意味着我们丢弃了协方差矩阵的第三行和第三列，将 3D 高斯压缩到 2D 图像平面。

**W 矩阵提取**：

```cuda
glm::mat3 W = glm::mat3(
    viewmatrix[0], viewmatrix[4], viewmatrix[8],
    viewmatrix[1], viewmatrix[5], viewmatrix[9],
    viewmatrix[2], viewmatrix[6], viewmatrix[10]);
```

**中间矩阵 T 和最终 2D 协方差**：

```cuda
glm::mat3 T = W * J;                        // T = W * J
glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
// 即: Sigma' = T^T * Sigma^T * T = (WJ)^T * Sigma * (WJ)
// 等价于论文中的 Sigma' = JW Sigma W^T J^T
```

> 注意代码中 $\boldsymbol{\Sigma}' = \mathbf{T}^\top \boldsymbol{\Sigma}^\top \mathbf{T}$，由于 $\boldsymbol{\Sigma}$ 对称，$\boldsymbol{\Sigma}^\top = \boldsymbol{\Sigma}$，因此等价于标准公式。

| 数学符号 | 物理意义 | 代码变量 |
|---|---|---|
| $\mathbf{W}$ | 视图矩阵的旋转部分（$3\times 3$） | `W` (glm::mat3) |
| $\mathbf{J}$ | 投影变换雅可比矩阵 | `J` (glm::mat3) |
| $\mathbf{T} = \mathbf{W}\mathbf{J}$ | 合成变换矩阵 | `T` (glm::mat3) |
| $\boldsymbol{\Sigma}$ | 3D 协方差矩阵 | `Vrk` (glm::mat3, 从 `cov3D` 构造) |
| $\boldsymbol{\Sigma}'$ | 2D 协方差矩阵（$2\times 2$） | `cov2D` (float3, 只存 a,b,c) |

### 1.3.4 低通滤波

为了保证每个高斯在图像上至少贡献一个像素（避免子像素走样），对 2D 协方差矩阵添加一个低通滤波：

```cuda
cov[0][0] += 0.3f;  // Sigma'_xx += 0.3
cov[1][1] += 0.3f;  // Sigma'_yy += 0.3
```

这相当于在 2D 高斯周围添加一个固定的模糊核，防止优化过程中高斯收缩到亚像素尺寸。

### 1.3.5 屏幕空间映射与裁剪

变换后的点被进一步投影到屏幕像素坐标，并进行边界裁剪（防止透视异常）：

```cuda
// 1. NDC裁剪: 限制在 tan_fov 的 1.3 倍范围内
const float limx = 1.3f * tan_fovx;
const float limy = 1.3f * tan_fovy;
const float txtz = t.x / t.z;
const float tytz = t.y / t.z;
t.x = min(limx, max(-limx, txtz)) * t.z;
t.y = min(limy, max(-limy, tytz)) * t.z;

// 2. 投影到齐次坐标
float4 p_hom = transformPoint4x4(p_orig, projmatrix);
float p_w = 1.0f / (p_hom.w + 0.0000001f);
float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

// 3. NDC 转像素坐标
float2 p_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
```

其中 `ndc2Pix` 的实现为：

```cuda
float ndc2Pix(float v, int S) {
    return ((v + 1.0) * S - 1.0) * 0.5;
}
```

这个公式将 NDC 坐标 $[-1, 1]$ 映射到像素坐标 $[0, S-1]$。

### 1.3.6 最终存储：conic_opacity

计算得到的 2D 协方差矩阵被转换为 **conic 矩阵**（即协方差矩阵的逆），与不透明度一起存储：

```cuda
// conic = Sigma'^{-1} = [[a, b], [b, c]]
// 对于 2x2 矩阵 [[A, B], [B, C]]，逆 = 1/(AC-B^2) * [[C, -B], [-B, A]]
float det = (a * c - b * b);
float conic_x = c / det;    // a'
float conic_y = -b / det;   // b'
float conic_z = a / det;    // c'

conic_opacity[idx] = { conic_x, conic_y, conic_z, opacities[idx] };
```

| 数学符号 | 物理意义 | 代码变量 |
|---|---|---|
| $\boldsymbol{\Sigma}'^{-1}$ | 2D 协方差逆矩阵（conic 矩阵） | `[conic_x, conic_y, conic_z]` = $\begin{pmatrix}a' & b' \\ b' & c'\end{pmatrix}$ |
| $o$ | 最终不透明度 | `opacity` / `conic_opacity.w` |

---

# 2. 可微光栅化：前向与反向传播

## 2.1 光栅化管线总览

3DGS 的可微光栅化器整体流程：

```
输入: P 个高斯，每个有 {mu, Sigma, o, SH系数}
  |
  +-- Step 1: Preprocess (每个高斯独立)
  |     +-- 视锥体裁剪 (frustum culling)
  |     +-- 计算3D到2D投影 (mu到mu', Sigma到Sigma')
  |     +-- SH 到 RGB 颜色计算
  |     +-- 计算高斯覆盖的 tile 范围
  |
  +-- Step 2: Tile Sorting (全局排序)
  |     +-- 每个高斯复制到其覆盖的所有 tile
  |     +-- 按 (tile_id, depth) 排序 -> 每个 tile 内深度有序
  |
  +-- Step 3: Render (每个 tile 独立并行)
        +-- Alpha blending 逐像素合成
```

## 2.2 Tile 划分与排序

将图像划分为 $16 \times 16$ 像素的 **tile**（代码中 `BLOCK_X = 16, BLOCK_Y = 16`）：

```cuda
dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
```

每个高斯通过其投影 $2\times 2$ 协方差矩阵计算一个 **bounding rectangle**，覆盖的每个 tile 都获得该高斯的一个副本：

```cuda
// 获取高斯在屏幕上的边界矩形
void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid) {
    rect_min = {
        max(0, (int)((p.x - max_radius) / BLOCK_X)),
        max(0, (int)((p.y - max_radius) / BLOCK_Y))
    };
    rect_max = {
        min(grid.x, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X)),
        min(grid.y, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y))
    };
}
```

其中 `max_radius` 由 2D 协方差矩阵的最大特征值决定（实际上是 `ceil(3 * sqrt(max_eigenval))` 的近似）。

每个高斯-tile 对生成一个**排序键**：

```cuda
uint64_t key = y * grid.x + x;  // tile_id (高位)
key <<= 32;
key |= *((uint32_t*)&depths[idx]);  // depth (低位)
```

然后使用 CUB 库的 `DeviceRadixSort::SortPairs` 对所有键值对进行基数排序。排序后，每个 tile 内的高斯按深度从小到大排列。

## 2.3 前向渲染公式

### 2.3.1 每个像素的合成

对于像素位置 $(x, y)$，遍历该 tile 内深度排序后的高斯列表，逐层合成颜色：

**步骤 1：计算高斯在该像素处的贡献值**（代码中的 `power`）：

$$
\mathcal{G}(\mathbf{d}) = \exp\left(-\frac{1}{2} \mathbf{d}^\top \boldsymbol{\Sigma}'^{-1} \mathbf{d}\right)
$$

其中 $\mathbf{d} = (x - \mu'_x, y - \mu'_y)^\top$ 是像素到投影高斯中心的偏移向量，$\boldsymbol{\Sigma}'^{-1}$ 是 conic 矩阵。

具体展开：

$$
\text{power} = -\frac{1}{2}\left(a' \cdot d_x^2 + c' \cdot d_y^2\right) - b' \cdot d_x d_y
$$

```cuda
float2 d = { xy.x - pixf.x, xy.y - pixf.y };
float4 con_o = collected_conic_opacity[j];
float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
```

**步骤 2：计算 alpha 值**：

$$
\alpha = \min\left(0.99,\; o \cdot \exp(\text{power})\right) = \min(0.99,\; o \cdot \mathcal{G}(\mathbf{d}))
$$

```cuda
float alpha = min(0.99f, con_o.w * exp(power));
```

这里用 $0.99$ 截断是为了数值稳定性（避免 $1-\alpha$ 恰好为 0 导致除零）。

**步骤 3：Alpha Blending**（论文公式 2-3）：

$$
\mathbf{C} = \sum_{i=1}^N \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)
$$

其中 $\mathbf{c}_i$ 是第 $i$ 个高斯的 RGB 颜色，$\alpha_i$ 是其透明度。

等价于**从前向后合成**的迭代形式：

$$
\begin{aligned}
T &\leftarrow 1 \\
\mathbf{C} &\leftarrow \mathbf{0} \\
\text{对每个高斯 } i \text{（从前向后）: } \\
&\quad \mathbf{C} \leftarrow \mathbf{C} + \mathbf{c}_i \cdot \alpha_i \cdot T \\
&\quad T \leftarrow T \cdot (1 - \alpha_i) \\
&\quad \text{如果 } T < \epsilon \text{，提前终止}
\end{aligned}
$$

```cuda
// 迭代过程
float T = 1.0f;
float C[CHANNELS] = { 0 };

for each Gaussian j in sorted order:
    float alpha = min(0.99f, con_o.w * exp(power));
    if (alpha < 1.0/255.0) continue;  // 跳过透明高斯
    float test_T = T * (1 - alpha);
    if (test_T < 0.0001f) break;       // 提前终止

    C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
    T = test_T;

// 叠加背景
out_color[pix_id] = C + T * bg_color;
```

**深度合成**：

$$
D = \sum_{i=1}^N d_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)
$$

其中 $d_i$ 是高斯在相机坐标系下的深度值（`depths[idx]`）。

| 数学符号 | 物理意义 | 代码变量 |
|---|---|---|
| $\mathcal{G}(\mathbf{d})$ | 2D 高斯在像素处的值 | `exp(power)` (float) |
| $o$ | 高斯不透明度 | `con_o.w` (float4的w分量) |
| $\alpha_i$ | 第 i 个高斯的混合权重 | `alpha` (float) |
| $T_i$ | 第 i 步的累积透射率 | `T` (float) |
| $\mathbf{c}_i$ | RGB 颜色 | `colors[global_id * C + ch]` |
| $\mathbf{C}$ | 累积像素颜色 | `C[ch]` (float[3]) |
| $d_i$ | 深度值（相机空间 z） | `depth[coll_id]` (float) |
| $D$ | 合成深度 | `D` (float) |

## 2.4 反向传播梯度推导

反向传播需要计算损失 $\mathcal{L}$ 对各个可学习参数（$\boldsymbol{\mu}, \boldsymbol{\Sigma}, o, \mathbf{c}$）的梯度。

### 2.4.1 渲染阶段梯度（renderCUDA backward）

从背景方向开始反向遍历高斯，使用**逆序链式法则**。已知每个像素的损失梯度 $\partial \mathcal{L} / \partial \mathbf{C}$（即 `dL_dpixels`），需要计算：

$$
\frac{\partial \mathcal{L}}{\partial \alpha_i}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{c}_i}, \quad \frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}'_i}, \quad \frac{\partial \mathcal{L}}{\partial \boldsymbol{\Sigma}'^{-1}_i}, \quad \frac{\partial \mathcal{L}}{\partial o_i}
$$

**步骤 1：恢复 $T_i$ 和累积量**

在反向传播中，从后向前恢复每个高斯的 $T_i$ 值和累积颜色：

```cuda
// 回到前向的起点
const float T_final = final_Ts[pix_id]; // 最终的 T
float T = T_final;

// 逆序遍历（从最后一个高斯开始）
for each Gaussian j in reverse order:
    T = T / (1.f - alpha);  // 恢复前一步的 T
```

**步骤 2：颜色梯度 $\partial \mathcal{L} / \partial \mathbf{c}_i$**

由合成公式 $\mathbf{C} = \sum \mathbf{c}_i \alpha_i T_i$，其中 $T_i = \prod_{j < i} (1 - \alpha_j)$：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{c}_i} = \frac{\partial \mathcal{L}}{\partial \mathbf{C}} \cdot \alpha_i T_i
$$

代码中对应的变量为 `dchannel_dcolor = alpha * T`。

**步骤 3：Alpha 梯度 $\partial \mathcal{L} / \partial \alpha_i$**

$\alpha_i$ 影响两部分：当前高斯的直接贡献 $\mathbf{c}_i \alpha_i T_i$，以及后续所有高斯和背景的权重。

这里的关键技巧是使用**累积剩余颜色** $\mathbf{C}_{\text{rem}}$：

$$
\mathbf{C}_{\text{rem}} = \underbrace{\sum_{k > i} \mathbf{c}_k \alpha_k \prod_{j=i+1}^{k-1} (1 - \alpha_j)}_{\text{后面高斯的合成}} + \underbrace{T_N \cdot \mathbf{c}_{\text{bg}}}_{\text{背景}}
$$

其中 $T_N = \prod_{j=1}^N (1 - \alpha_j)$ 是最终透射率。

像素颜色 $\mathbf{C} = \sum_{k=1}^N \mathbf{c}_k \alpha_k T_k + T_{N+1} \cdot \mathbf{c}_{\text{bg}}$。

对 $\alpha_i$ 求导：

$$
\frac{\partial \mathbf{C}}{\partial \alpha_i} = \mathbf{c}_i T_i - T_i \underbrace{\left( \sum_{k > i} \mathbf{c}_k \alpha_k \frac{T_k}{T_{i+1}} + \frac{T_{N+1}}{T_{i+1}} \mathbf{c}_{\text{bg}} \right)}_{\text{累积剩余颜色 } \mathbf{C}_{\text{rem}}}
$$

注意 $\frac{T_k}{T_{i+1}} = \prod_{j=i+1}^{k-1} (1 - \alpha_j)$ 就是从 $i+1$ 开始的透射率，所以括号内正是从第 $i+1$ 个高斯开始合成到背景的总颜色 $\mathbf{C}_{\text{rem}}$。

整理得：

$$
\frac{\partial \mathcal{L}}{\partial \alpha_i} = \frac{\partial \mathcal{L}}{\partial \mathbf{C}} \cdot \left( (\mathbf{c}_i - \mathbf{C}_{\text{rem}}) \cdot T_i \right)
$$

代码中的实现：

```cuda
// 累加剩余颜色（从后向前遍历时自然累积）
for (int ch = 0; ch < C; ch++) {
    accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
    last_color[ch] = c;

    dL_dalpha += (c - accum_rec[ch]) * dL_dpixel[ch];
}
dL_dalpha *= T;  // 乘上 T_i
```

其中 `accum_rec[ch]` 正是 $\mathbf{C}_{\text{rem}}$，因为逆序遍历时已经处理了后面所有高斯的贡献。

还需要考虑背景的贡献：

```cuda
float bg_dot_dpixel = 0;
for (int i = 0; i < C; i++)
    bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
```

**步骤 4：梯度传播到高斯值 $\mathcal{G}$ 和不透明度 $o$**

由 $\alpha = \min(0.99, o \cdot \mathcal{G})$，且通常 $o \cdot \mathcal{G} < 0.99$：

$$
\frac{\partial \mathcal{L}}{\partial \mathcal{G}} = o \cdot \frac{\partial \mathcal{L}}{\partial \alpha}, \quad
\frac{\partial \mathcal{L}}{\partial o} = \mathcal{G} \cdot \frac{\partial \mathcal{L}}{\partial \alpha}
$$

**步骤 5：梯度传播到 2D 均值 $\boldsymbol{\mu}'$**

高斯值 $\mathcal{G}$ 通过 $\boldsymbol{\mu}'$ 影响损失：

$$
\mathcal{G}(\mathbf{d}) = \exp\left(-\frac{1}{2} \mathbf{d}^\top \boldsymbol{\Sigma}'^{-1} \mathbf{d}\right), \quad \mathbf{d} = \mathbf{p} - \boldsymbol{\mu}'
$$

对 $\boldsymbol{\mu}'$ 求导：

$$
\frac{\partial \mathcal{G}}{\partial \boldsymbol{\mu}'} = \mathcal{G} \cdot \boldsymbol{\Sigma}'^{-1} \mathbf{d}
$$

具体分量为：

```cuda
const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;  // dG/dmu'_x
const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;  // dG/dmu'_y
```

乘以像素坐标到 NDC 的缩放因子：

```cuda
atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
```

其中 `ddelx_dx = 0.5 * W`, `ddely_dy = 0.5 * H`。

**步骤 6：梯度传播到 conic 矩阵 $\boldsymbol{\Sigma}'^{-1}$**

$$
\frac{\partial \mathcal{G}}{\partial a'} = -\frac{1}{2} \mathcal{G} \cdot d_x^2, \quad
\frac{\partial \mathcal{G}}{\partial b'} = -\frac{1}{2} \mathcal{G} \cdot d_x d_y, \quad
\frac{\partial \mathcal{G}}{\partial c'} = -\frac{1}{2} \mathcal{G} \cdot d_y^2
$$

```cuda
atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);  // dL/da'
atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);  // dL/db'
atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);  // dL/dc'
```

### 2.4.2 Preprocess 阶段梯度

将渲染阶段的梯度继续反向传播到 3D 参数：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}} \quad \frac{\partial \mathcal{L}}{\partial \mathbf{s}} \quad \frac{\partial \mathcal{L}}{\partial \mathbf{q}} \quad \frac{\partial \mathcal{L}}{\partial \mathbf{c}_{\text{SH}}}
$$

**2D 均值反向传播到 3D 均值**：

$\boldsymbol{\mu}'$ 是通过投影矩阵 $\mathbf{P}$ 从 $\boldsymbol{\mu}$ 得到的：

$$
\boldsymbol{\mu}'_h = \mathbf{P} \cdot \boldsymbol{\mu}_h, \quad \boldsymbol{\mu}' = (\mu'_h.x / \mu'_h.w,\; \mu'_h.y / \mu'_h.w)
$$

梯度传播（代码中 `preprocessCUDA` 的反向部分）：

```cuda
float4 m_hom = transformPoint4x4(m, proj);
float m_w = 1.0f / (m_hom.w + 0.0000001f);

float mul1 = (proj[0]*m.x + proj[4]*m.y + proj[8]*m.z + proj[12]) * m_w * m_w;
float mul2 = (proj[1]*m.x + proj[5]*m.y + proj[9]*m.z + proj[13]) * m_w * m_w;

dL_dmean.x = (proj[0]*m_w - proj[3]*mul1) * dL_dmean2D.x + (proj[1]*m_w - proj[3]*mul2) * dL_dmean2D.y;
dL_dmean.y = (proj[4]*m_w - proj[7]*mul1) * dL_dmean2D.x + (proj[5]*m_w - proj[7]*mul2) * dL_dmean2D.y;
dL_dmean.z = (proj[8]*m_w - proj[11]*mul1) * dL_dmean2D.x + (proj[9]*m_w - proj[11]*mul2) * dL_dmean2D.y;
```

**2D 协方差反向传播到 3D 协方差**：

通过链式法则经过 $\boldsymbol{\Sigma}' = \mathbf{J}\mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top\mathbf{J}^\top$ 链，具体实现见 `computeCov2DCUDA` 的反向部分。该 kernel 将 `dL_dconic` 传播到 `dL_dmean3D` 和 `dL_dcov`。

**3D 协方差反向传播到 scale 和 rotation**：

由 $\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S}^2 \mathbf{R}^\top$，将梯度从 $\boldsymbol{\Sigma}$ 传播到旋转矩阵 $\mathbf{R}$（进而到四元数 $\mathbf{q}$）和缩放矩阵 $\mathbf{S}$（进而到 $\mathbf{s}$）。

**颜色反向传播到 SH 系数**：

`computeColorFromSH` 的反向版本（在 `backward.cu` 中）将颜色梯度通过 SH 求导传播回 SH 系数和视角方向（即 3D 均值位置）。

**梯度传播总链**：

```
dL/dC_out (像素)
  -> dL/dalpha_i, dL/dc_i           (renderCUDA backward)
    -> dL/do_i                       (透明度梯度)
    -> dL/dG_i -> dL/dmu'_i          (2D 均值梯度)
    -> dL/dSigma'^{-1}_i             (conic 梯度)
      -> dL/dSigma'                  (逆矩阵梯度)
        -> dL/dSigma                 (computeCov2DCUDA backward)
          -> dL/ds, dL/dq            (computeCov3D backward)
        -> dL/dmu_{3D}               (投影的均值梯度)
    -> dL/dc_i -> dL/dSH, dL/dmu_{3D}  (SH 反向)
```

---

# 3. 密度控制数学逻辑

## 3.1 概述

3DGS 的核心创新之一是**自适应密度控制 (Adaptive Density Control)**，在学习过程中动态调整高斯的数量、位置和形状。控制逻辑每 $N$ 个迭代执行一次（默认 $N=100$）。

整个控制流程分为两大阶段：

| 阶段 | 迭代范围 | 操作 |
|---|---|---|
| 热身阶段 | $0 \sim 500$ 步 | Adam 优化参数，仅做不透明度重置 |
| 密度控制阶段 | $> 500$ 步 | 每 100 步执行：筛选 -> 克隆/分裂 -> 剔除 |

## 3.2 筛选标准

每个高斯的"重要性"通过其位置梯度 $\nabla_{\boldsymbol{\mu}} \mathcal{L}$ 的累积量来衡量。

**视图空间位置梯度**：

对每个高斯，梯度来自反向传播中的 `dL_dmean3D`。我们需要这些梯度的绝对值累积：

$$
\boldsymbol{\tau}_i^{\text{(accum)}} \mathrel{+}= \|\nabla_{\boldsymbol{\mu}_i} \mathcal{L}\|_2 \quad \text{（每次反向传播后累加）}
$$

同时累加该高斯被处理的次数（即在多少个视图中可见）：

$$
c_i \mathrel{+}= 1 \quad \text{（每个视图累加）}
$$

**平均梯度**：

$$
\bar{\tau}_i = \frac{\boldsymbol{\tau}_i^{\text{(accum)}}}{c_i}
$$

**筛选条件**：

当 $\bar{\tau}_i > \tau_{\text{thresh}}$（默认 $= 0.0002$）时，该高斯被认为"需要分裂或克隆"。

## 3.3 克隆与分裂

根据被标记高斯的大小（即缩放 $\mathbf{s}$）决定操作类型：

### 3.3.1 克隆（小高斯）

若 $s_{\text{max}} = \max(s_x, s_y, s_z) \leq \epsilon_{\text{size}}$：

- 创建一个新的高斯，其参数与原高斯**完全相同**
- 将原高斯和新高斯的均值沿梯度方向**微移**：

$$
\boldsymbol{\mu}_{\text{orig}} \leftarrow \boldsymbol{\mu}_{\text{orig}} + \epsilon \cdot \nabla_{\boldsymbol{\mu}} \mathcal{L} / \|\nabla_{\boldsymbol{\mu}} \mathcal{L}\|
$$

$$
\boldsymbol{\mu}_{\text{new}} \leftarrow \boldsymbol{\mu}_{\text{new}} - \epsilon \cdot \nabla_{\boldsymbol{\mu}} \mathcal{L} / \|\nabla_{\boldsymbol{\mu}} \mathcal{L}\|
$$

其中 $\epsilon$ 是一个小量（通常为 `0.01 * 场景范围`）。

**物理意义**：小高斯表示场景中细节丰富、梯度信号强的区域，应该增加密度以捕获更多细节。

### 3.3.2 分裂（大高斯）

若 $s_{\text{max}} > \epsilon_{\text{size}}$：

- 将原高斯分裂为两个更小的高斯
- 缩放减半：$s_{\text{new}} = s_{\text{orig}} / 1.6$

$$
\begin{aligned}
\mathbf{s}_{\text{new,1}} &= \mathbf{s}_{\text{orig}} / 1.6 \\
\mathbf{s}_{\text{new,2}} &= \mathbf{s}_{\text{orig}} / 1.6
\end{aligned}
$$

- 均值位置使用原始高斯的采样：

$$
\boldsymbol{\mu}_{\text{new,1}} = \boldsymbol{\mu}_{\text{orig}} + \delta \cdot \mathbf{u}_1 \cdot s_{\text{max}} / 1.6
$$

$$
\boldsymbol{\mu}_{\text{new,2}} = \boldsymbol{\mu}_{\text{orig}} - \delta \cdot \mathbf{u}_1 \cdot s_{\text{max}} / 1.6
$$

其中 $\mathbf{u}_1$ 是协方差矩阵最大特征值对应的特征向量方向，$\delta$ 是采样偏移量。

> **关键区别**：克隆增加场景中高斯的数量（密度增加），而分裂用两个小高斯替换一个大高斯（数量不变但分辨率提高）。

## 3.4 不透明度重置

每经过一段迭代（通常每 3000 步），对所有不透明度接近于 0 的高斯重置其不透明度：

$$
o_i^{\text{(new)}} = \sigma(\text{raw\_opacity}_{\text{init}})
$$

并在重置后将其缩放缩小到原来的 1/10：

$$
\mathbf{s}_i^{\text{(new)}} = \mathbf{s}_i \times 0.1
$$

**物理意义**：防止 GS 陷入"floaters"局部最优，即一堆透明高斯聚集在某些区域但没有实际贡献。通过重置不透明度，给这些高斯重新学习的机会。

**代码逻辑**（Python 层 `scene/gaussian_model.py` 中的 `prune_and_densify`）：

```python
# 不透明度重置
if iter > opt.opacity_reset_interval // 2 and iter % opt.opacity_reset_interval == 0:
    self.opacities.data = torch.sigmoid(self.opacities_orig)
    self.scales.data = self.scales_orig

# 或对特定高斯重置
reset_opacities_mask = (self.opacities.data < opt.opacity_threshold).squeeze()
self.opacities[reset_opacities_mask] = torch.sigmoid(self.opacities_orig[reset_opacities_mask]) / 10
```

## 3.5 剔除策略

每轮密度控制还包含剔除步骤：

**1. 基于不透明度剔除**（每 100 步）：

$$
\text{remove } i \text{ if } o_i < \epsilon_\alpha \quad (\epsilon_\alpha = 0.005)
$$

**2. 基于体积剔除**（每 100 步）：

$$
\text{remove } i \text{ if } s_{x,i} \cdot s_{y,i} \cdot s_{z,i} > \epsilon_{\text{vol}}
$$

即剔除过大的高斯（通常是"floaters"或未收敛的冗余高斯）。

**3. 基于可见性剔除**（每 100 步）：

如果一个高斯在一段时间内（训练窗口内）没有被任何视图观察到（touches 计数为 0），则将其剔除。

## 3.6 优化目标与损失函数

3DGS 使用混合损失函数进行端到端训练：

$$
\mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}
$$

其中：

- $\mathcal{L}_1$ 是渲染图像与 GT 图像的像素级 L1 损失
- $\mathcal{L}_{\text{D-SSIM}}$ 是结构相似性损失（SSIM 的差异部分：$(1 - \text{SSIM})/2$）
- $\lambda = 0.2$（默认）

**优化器**：Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-15}$)

**学习率**：
- 位置 $\boldsymbol{\mu}$ 的学习率：初始 `lr = 1.6e-4`，指数衰减到 `1.6e-6`
- 其他参数各有独立学习率

## 3.7 密度控制完整算法总结

```
Algorithm: Adaptive Density Control (每100步执行)

Input: 高斯集合 G = {g_i = (mu_i, Sigma_i, o_i, SH_i)}
参数: tau_thresh=0.0002, alpha_thresh=0.005, 密度控制起始=500步

1. 对每个高斯 g_i:
   - 计算平均位置梯度 tau_bar_i = ||grad_position||_accum / view_count_i

2. 筛选: G_selected = {g_i | tau_bar_i > tau_thresh}

3. 克隆/分裂:
   For each g_i in G_selected:
     if max(s_i) <= size_threshold:    // 小高斯 -> 克隆
       g_clone = copy(g_i)
       偏移 mu_i 和 mu_clone 沿梯度方向相反移动
       G <- G U {g_clone}
     else:                              // 大高斯 -> 分裂
       创建两个新高斯，scales减半
       沿最大特征向量方向偏移均值
       移除原高斯 g_i
       G <- G U {g_new1, g_new2}

4. 剔除:
   - 移除 o_i < alpha_thresh 的高斯
   - 移除体积过大的高斯
   - 移除长时不可见的高斯

5. (每3000步) 不透明度重置:
   - 重置 proximity=0 的高斯的不透明度
```

---

## 附录A：代码变量名速查表

| 数学符号 | 代码变量（CUDA） | 代码变量（Python） | 维度 | 说明 |
|---|---|---|---|---|
| $\boldsymbol{\mu}$ | `means3D`, `orig_points` | `means3D` | [P, 3] | 3D 高斯均值 |
| $\mathbf{s}$ | `scales` | `scales` | [P, 3] | 缩放（log 空间） |
| $\mathbf{q}$ | `rotations` | `rotations` | [P, 4] | 旋转四元数 |
| $\boldsymbol{\Sigma}$ | `cov3D` | `cov3D` | [P, 6] | 3D 协方差（上三角） |
| $\boldsymbol{\mu}'$ | `means2D`, `points_xy_image` | `means2D` | [P, 2] | 投影到屏幕的 2D 均值 |
| $\boldsymbol{\Sigma}'^{-1}$ | `conic_opacity.xyz` | -- | [P, 4] | 逆 2D 协方差 (a,b,c) |
| $o$ | `conic_opacity.w`, `opacities` | `opacities` | [P, 1] | 不透明度（sigmoid后） |
| $\mathbf{c}$ | `rgb`, `colors` | -- | [P, 3] | SH 转换后的 RGB |
| $\mathbf{c}_{\text{SH}}$ | `shs` | `features` | [P, (D+1)^2, 3] | 球谐系数 |
| $T$ | `final_T` | -- | [H*W] | 累积透射率 |
| $C$ | `out_color` | -- | [3, H*W] | 合成像素颜色 |
| $D$ | `out_depth` | `depth` | [H*W] | 合成深度 |
| $\nabla_{\boldsymbol{\mu}}\mathcal{L}$ | `dL_dmeans` | `xyz_gradient_accum` | [P, 3] | 位置梯度累积 |
| $\bar{\tau}_i$ | -- | `grad_norm / view_count` | [P] | 平均位置梯度范数 |

## 附录B：关键论文引用

| 引用内容 | 出处 |
|---|---|
| EWA Splatting 投影公式 | Zwicker et al., "EWA Splatting", IEEE TVCG 2002 |
| 3DGS 原始论文 | Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023 |
| 可微光栅化框架 | 基于 Laine et al., "Modular Primitives for High-Performance Differentiable Rendering", SIGGRAPH 2020 |
| 球谐函数表示视角相关颜色 | Zhang et al., "Differentiable Point-Based Radiance Fields for Efficient View Synthesis", SIGGRAPH 2022 |

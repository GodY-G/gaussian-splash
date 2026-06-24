// ============================================================================
// 3D Gaussian Splatting — 前向渲染 CUDA 核函数 中文详细注释版
// ============================================================================
//
// 原始代码来源:
//   "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
//   Kerbl, Kopanas, Leimkühler, Drettakis (SIGGRAPH 2023)
//   GRAPHDECO research group, Inria
//
// 本文件在原始 forward.cu 基础上增加逐行中文注释，阐明每个变量、
// 每一步的图形学意义和并行逻辑。
//
// ============================================================================
// 一、整体前向渲染流程概览
// ============================================================================
//
// 整个前向渲染分为两大 CUDA kernel + 若干 CPU 侧辅助步骤，
// 由 Rasterizer::forward() (rasterizer_impl.cu) 编排：
//
//  ┌─────────────────────────────────────────────────────────────────────┐
//  │ Step 1: Preprocess Kernel (每个 Gaussian 一个线程)                  │
//  │  - 视锥体裁剪（剔除不在视锥内的 Gaussian）                          │
//  │  - 世界坐标 → 屏幕投影变换                                         │
//  │  - 计算 3D → 2D 协方差矩阵（EWA Splatting 公式 29/31）             │
//  │  - 计算屏幕空间 bounding box (瓦片级别的矩形范围)                   │
//  │  - 球谐函数 → RGB 颜色转换                                         │
//  │  - 输出: means2D, depths, conic_opacity, radii, tiles_touched 等    │
//  └─────────────────────────────────────────────────────────────────────┘
//                          │
//                          ▼
//  ┌─────────────────────────────────────────────────────────────────────┐
//  │ Step 2: Inclusive Prefix Sum (CUB 库)                              │
//  │  - 对 tiles_touched 数组做前缀和                                   │
//  │  - 最后一个元素 = 所有 Gaussian 的瓦片实例总数 num_rendered        │
//  └─────────────────────────────────────────────────────────────────────┘
//                          │
//                          ▼
//  ┌─────────────────────────────────────────────────────────────────────┐
//  │ Step 3: duplicateWithKeys Kernel (每个 Gaussian 一个线程)           │
//  │  - 对每个 Gaussian，遍历其覆盖的所有瓦片                           │
//  │  - 为每个(Gaussian, 瓦片)对生成 key/value                          │
//  │  - key = (tile_id << 32) | depth_bits                              │
//  │    value = Gaussian 索引 idx                                       │
//  └─────────────────────────────────────────────────────────────────────┘
//                          │
//                          ▼
//  ┌─────────────────────────────────────────────────────────────────────┐
//  │ Step 4: Radix Sort (CUB 库)                                        │
//  │  - 按 key 对所有实例进行基数排序                                    │
//  │  - 排序后: 同瓦片内 Gaussian 排在一起且按深度递增                   │
//  └─────────────────────────────────────────────────────────────────────┘
//                          │
//                          ▼
//  ┌─────────────────────────────────────────────────────────────────────┐
//  │ Step 5: identifyTileRanges Kernel (每个实例一个线程)                │
//  │  - 扫描排序后的 keys，找出每个瓦片的起止索引                       │
//  │  - 写入 ranges[tile_id] = {start, end}                             │
//  └─────────────────────────────────────────────────────────────────────┘
//                          │
//                          ▼
//  ┌─────────────────────────────────────────────────────────────────────┐
//  │ Step 6: Render Kernel (每个瓦片一个 block, 每个像素一个线程)        │
//  │  - 协作加载 Gaussian 到 shared memory                              │
//  │  - Alpha Blending: C = Σ(c_i * α_i * T_i)                         │
//  │  - 前向传播 alpha 合成即论文公式 (2)(3)                            │
//  └─────────────────────────────────────────────────────────────────────┘
//
// ============================================================================
// 二、瓦片划分 (Tile Partitioning) 的并行逻辑
// ============================================================================
//
// 图像被划分为 BLOCK_X × BLOCK_Y 像素大小的瓦片，
// 默认 BLOCK_X=16, BLOCK_Y=16，即 16×16=256 像素/瓦片。
//
// 瓦片网格维度: grid = (ceil(W/16), ceil(H/16))
//
// 每个 CUDA block 负责一个瓦片，256 个线程各负责一个像素。
//   1. 同瓦片内所有线程共享同一组 Gaussian 数据 (shared memory)
//   2. 瓦片间完全独立，天然并行
//   3. 每个线程只需计算自己对应的一个像素
//
// Gaussian 到瓦片的映射: 椭圆投影 → bounding box → 覆盖的瓦片集。
//
// ============================================================================
// 三、高斯排序 (Gaussian Sorting) 策略
// ============================================================================
//
// 排序 key = (tile_id << 32) | float_bits(depth)
//
// tile_id = tile_y * grid_x + tile_x (一维瓦片索引)
// depth = Gaussian 中心在相机空间的 Z 值
//
// float 的位模式直接用 reinterpret_cast<uint32_t> 读取，
// 按 uint64_t 排序后，高位 tile_id 保证瓦片聚集，
// 低位 depth 保证瓦片内按深度递增 (由近到远)。
//
// ============================================================================
// 四、Alpha 合成公式
// ============================================================================
//
// Front-to-back alpha blending:
//
//   C = Σ_{i=1..N} c_i × α_i × Π_{j=1..i-1} (1 - α_j)
//   T = Π_{j=1..N} (1 - α_j)   // 最终透射率
//
// 其中:
//   c_i = RGB 颜色（来自球谐函数或预计算）
//   α_i = o_i × exp(-0.5 × Δ^T × Σ^{-1} × Δ)
//   o_i = 学习的 opacity,  Σ^{-1} = conic matrix
//   Δ = 像素到 Gaussian 中心的 2D 偏移
//
// 提前终止:
//   1. α_i < 1/255（肉眼不可见）
//   2. T × (1 - α_i) < 0.0001（后续贡献 < 0.01%）
//
// ============================================================================

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// ============================================================================
// computeColorFromSH — 球谐函数 → RGB 颜色转换
// ============================================================================
// 输入:
//   idx        — Gaussian 索引
//   deg        — 球谐阶数 (0~3)
//   max_coeffs — 每个 Gaussian 的 SH 系数数量
//   means      — 所有 Gaussian 的 3D 均值
//   campos     — 相机位置
//   shs        — 球谐系数数组 [P × max_coeffs × 3]
//   clamped    — 记录哪些通道被钳位（用于反向传播）
// 输出:
//   返回 RGB 颜色 (glm::vec3)
//
// 物理意义:
//   球谐函数将视角方向映射为颜色。方向向量 (x,y,z) 从 Gaussian 中心
//   指向相机，带入 SH 基函数计算颜色。
//   SH_C0~SH_C3 是归一化常数（Legendre 多项式系数）。
// ============================================================================
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs,
    const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
    // 实现参考:
    // "Differentiable Point-Based Radiance Fields for Efficient View Synthesis"
    // Zhang et al. (2022)

    // Step 1: 计算从 Gaussian 中心指向相机的方向向量
    // pos: 当前 Gaussian 的世界坐标 (x, y, z)
    glm::vec3 pos = means[idx];
    // dir: 从 Gaussian 中心指向相机
    glm::vec3 dir = pos - campos;
    // 归一化到单位向量 (SH 基函数只依赖方向)
    dir = dir / glm::length(dir);

    // Step 2: 定位当前 Gaussian 的 SH 系数起始位置
    // shs 布局: [G0_coeff0_RGB, G0_coeff1_RGB, ..., G1_coeff0_RGB, ...]
    glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

    // Step 3: 从最低阶 (l=0) 开始累加
    // l=0: 直流分量 (DC)，与方向无关，相当于平均颜色
    glm::vec3 result = SH_C0 * sh[0];

    // l=1: 3 个系数，对应偶极子分布
    if (deg > 0)
    {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;

        // SH 基函数: Y_1^{-1} ∝ y, Y_1^0 ∝ z, Y_1^1 ∝ x
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        // l=2: 5 个系数，对应四极子分布
        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                SH_C2[0] * xy * sh[4] +      // Y_2^{-2} ∝ xy
                SH_C2[1] * yz * sh[5] +      // Y_2^{-1} ∝ yz
                SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +  // Y_2^0 ∝ 3z^2-1
                SH_C2[3] * xz * sh[7] +      // Y_2^1 ∝ xz
                SH_C2[4] * (xx - yy) * sh[8]; // Y_2^2 ∝ x^2-y^2

            // l=3: 7 个系数
            if (deg > 2)
            {
                result = result +
                    SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                    SH_C3[1] * xy * z * sh[10] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                    SH_C3[5] * z * (xx - yy) * sh[14] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }

    // Step 4: 加 0.5 偏移并钳位到非负
    // 球谐函数值域对称，加 0.5 将结果移到 [0, 1] 范围
    result += 0.5f;

    // 记录被钳位的通道（反向传播需要知道哪些梯度被截断）
    clamped[3 * idx + 0] = (result.x < 0);
    clamped[3 * idx + 1] = (result.y < 0);
    clamped[3 * idx + 2] = (result.z < 0);
    return glm::max(result, 0.0f);
}

// ============================================================================
// computeCov2D — 计算 2D 屏幕空间协方差矩阵
// ============================================================================
// 输入:
//   mean     — Gaussian 3D 世界坐标
//   focal_x  — X 方向焦距 (像素)
//   focal_y  — Y 方向焦距 (像素)
//   tan_fovx — X 方向视场角正切
//   tan_fovy — Y 方向视场角正切
//   cov3D    — 3D 世界空间协方差 (6 元素)
//   viewmatrix — 4×4 视图矩阵 (列主序)
// 输出:
//   float3 {Σ_xx, Σ_xy, Σ_yy} (对称矩阵上三角)
//
// 原理: EWA Splatting (Zwicker et al., 2002) 公式 29/31
//   Σ_2D = (W J) Σ_3D (W J)^T
//   J = 投影变换的 Jacobian (透视投影的线性近似)
//   W = 视图矩阵的旋转部分 (3×3)
// ============================================================================
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y,
    float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
    // Step 1: 将 Gaussian 中心变换到相机空间 (view space)
    float3 t = transformPoint4x3(mean, viewmatrix);
    // t.z = 深度 (View 空间 Z 值)

    // Step 2: 对近平面外的点做 clamping，防止数值不稳定
    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;   // 透视除法后 NDC x
    const float tytz = t.y / t.z;   // 透视除法后 NDC y
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    // Step 3: 计算投影 Jacobian J
    // J = [ ∂u/∂x  ∂u/∂y  ∂u/∂z ]   u = focal_x * x/z
    //     [ ∂v/∂x  ∂v/∂y  ∂v/∂z ]   v = focal_y * y/z
    //     [   0      0      0    ]
    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

    // Step 4: 取视图矩阵旋转部分 W (3×3)
    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    // Step 5: T = W * J (合成变换矩阵)
    glm::mat3 T = W * J;

    // Step 6: 构建 3D 协方差矩阵 Vrk
    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    // Step 7: Σ_2D = T Σ_3D T^T
    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    // Step 8: 低通滤波，保证每个 Gaussian 至少 1 像素宽/高
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;

    // 返回上三角 3 个元素 (第 3 行/列投影后为 0)
    return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// ============================================================================
// computeCov3D — 从缩放和旋转参数计算 3D 协方差矩阵
// ============================================================================
// 输入:
//   scale   — 三轴缩放 (s_x, s_y, s_z)
//   mod     — 全局缩放调节因子
//   rot     — 单位四元数 (r, x, y, z)
//   cov3D   — 输出: 6 元素协方差 [Σ_xx, Σ_xy, Σ_xz, Σ_yy, Σ_yz, Σ_zz]
//
// 原理: Σ = (S R)^T (S R) = R^T S^T S R
//   S = diag(s_x, s_y, s_z) — 对角缩放
//   R = 四元数 → 旋转矩阵
//   物理意义: 协方差矩阵定义 Gaussian 椭球的形状，
//   特征向量 = 椭球主轴方向，特征值 = 主轴长度。
// ============================================================================
__device__ void computeCov3D(const glm::vec3 scale, float mod,
    const glm::vec4 rot, float* cov3D)
{
    // Step 1: 构建缩放矩阵 S
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    // Step 2: 四元数 → 旋转矩阵
    glm::vec4 q = rot;
    float r = q.x;  // 实部
    float x = q.y;  // 虚部 i
    float y = q.z;  // 虚部 j
    float z = q.w;  // 虚部 k

    // 标准公式: R = [[1-2(y²+z²), 2(xy-rz),   2(xz+ry)  ],
    //                [2(xy+rz),   1-2(x²+z²), 2(yz-rx)  ],
    //                [2(xz-ry),   2(yz+rx),   1-2(x²+y²)]]
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    // Step 3: M = S R (合成缩放 + 旋转)
    glm::mat3 M = S * R;

    // Step 4: Σ = M^T M (保证对称正定)
    glm::mat3 Sigma = glm::transpose(M) * M;

    // Step 5: 保存对称矩阵上三角 6 元素
    cov3D[0] = Sigma[0][0];  // Σ_xx
    cov3D[1] = Sigma[0][1];  // Σ_xy
    cov3D[2] = Sigma[0][2];  // Σ_xz
    cov3D[3] = Sigma[1][1];  // Σ_yy
    cov3D[4] = Sigma[1][2];  // Σ_yz
    cov3D[5] = Sigma[2][2];  // Σ_zz
}

// ============================================================================
// preprocessCUDA — 前处理核函数（每个 Gaussian 一个线程）
// ============================================================================
// 光栅化前的准备阶段。每个 CUDA 线程处理一个 Gaussian，
// 完成: 视锥裁剪 → 投影 → 协方差 → 颜色 → 瓦片范围。
//
// 模板参数 C: 颜色通道数 (默认 3: RGB)
// 线程块大小 = 256, 网格大小 = ceil(P/256)
// 每个线程独立处理一个 Gaussian，天然数据并行。
// ============================================================================
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
    const float* orig_points,          // 输入: Gaussian 3D 坐标 [P×3]
    const glm::vec3* scales,           // 输入: 缩放参数 [P×3]
    const float scale_modifier,        // 输入: 全局缩放调节因子
    const glm::vec4* rotations,        // 输入: 四元数旋转 [P×4]
    const float* opacities,            // 输入: 不透明度 [P]
    const float* shs,                  // 输入: 球谐系数 [P×M×3]
    bool* clamped,                     // 输出: 颜色钳位标记 [P×3]
    const float* cov3D_precomp,        // 输入: 预计算 3D 协方差 [P×6] (可选)
    const float* colors_precomp,       // 输入: 预计算颜色 [P×3] (可选)
    const float* viewmatrix,           // 输入: 4×4 视图矩阵
    const float* projmatrix,           // 输入: 4×4 投影矩阵
    const glm::vec3* cam_pos,          // 输入: 相机位置
    const int W, int H,                // 输入: 图像宽高
    const float tan_fovx, float tan_fovy,  // 输入: 视场角正切
    const float focal_x, float focal_y,    // 输入: 焦距 (像素)
    int* radii,                        // 输出: 屏幕空间半径 [P]
    float2* points_xy_image,           // 输出: 屏幕空间 2D 坐标 [P]
    float* depths,                     // 输出: 相机空间深度 [P]
    float* cov3Ds,                     // 输出: 3D 协方差 [P×6]
    float* rgb,                        // 输出: RGB 颜色 [P×3]
    float4* conic_opacity,             // 输出: {conic.x, conic.y, conic.z, opacity} [P]
    const dim3 grid,                   // 输入: 瓦片网格维度
    uint32_t* tiles_touched,           // 输出: 每个 Gaussian 覆盖的瓦片数 [P]
    bool prefiltered)                  // 输入: 是否已预过滤
{
    // Step 0: 获取全局线程索引，超出总数则退出
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Step 1: 初始化为 0 (视锥外的 Gaussian 不会被处理)
    radii[idx] = 0;
    tiles_touched[idx] = 0;

    // Step 2: 视锥体裁剪
    // in_frustum: 检查 Gaussian 中心是否在视锥内
    // 条件: 深度 z > 0.2 (在相机前方)
    //       NDC 坐标在 [-1.3, 1.3] 范围内
    float3 p_view;
    if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
        return;

    // Step 3: 投影变换 (世界 → 裁剪空间 → NDC → 像素)
    float3 p_orig = {
        orig_points[3 * idx],
        orig_points[3 * idx + 1],
        orig_points[3 * idx + 2]
    };
    // 4×4 齐次变换到裁剪空间
    float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    // 透视除法: 齐次 → NDC (加 1e-7 防除零)
    float p_w = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    // Step 4: 计算 3D 协方差矩阵
    const float* cov3D;
    if (cov3D_precomp != nullptr)
    {
        cov3D = cov3D_precomp + idx * 6;
    }
    else
    {
        computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
        cov3D = cov3Ds + idx * 6;
    }

    // Step 5: 计算 2D 屏幕空间协方差
    float3 cov = computeCov2D(p_orig, focal_x, focal_y,
                              tan_fovx, tan_fovy, cov3D, viewmatrix);

    // Step 6: 求 2D 协方差逆矩阵 (conic matrix)
    // Σ = [a  b]   逆 = 1/det × [c  -b]
    //     [b  c]              [-b  a]
    // det = a*c - b²
    float det = (cov.x * cov.z - cov.y * cov.y);
    if (det == 0.0f)
        return;
    float det_inv = 1.f / det;
    // conic = {c/det, -b/det, a/det}
    float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

    // Step 7: 计算屏幕空间 bounding box
    // 通过特征值确定椭圆主轴: λ = mid ± sqrt(mid² - det)
    // mid = (a + c) / 2
    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));  // 最大特征值
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));  // 最小特征值
    // 3σ 半径覆盖 99.7% Gaussian 能量
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

    // NDC → 像素坐标
    float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

    // 像素半径 → 瓦片级别 bounding box
    uint2 rect_min, rect_max;
    getRect(point_image, my_radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    // Step 8: 计算颜色 (如果未预计算则从 SH 转换)
    if (colors_precomp == nullptr)
    {
        glm::vec3 result = computeColorFromSH(
            idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
        rgb[idx * C + 0] = result.x;
        rgb[idx * C + 1] = result.y;
        rgb[idx * C + 2] = result.z;
    }

    // Step 9: 输出后续渲染所需数据
    depths[idx] = p_view.z;                    // 相机空间深度 (排序用)
    radii[idx] = my_radius;                    // 屏幕空间半径
    points_xy_image[idx] = point_image;        // 屏幕像素坐标
    // conic_opacity 打包: xyz = conic, w = opacity
    conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// ============================================================================
// renderCUDA — 主渲染核函数（每个瓦片一个 block，每个像素一个线程）
// ============================================================================
// 这是整个前向渲染的核心。每个 CUDA block 负责一个 16×16 瓦片，
// 256 个线程各负责一个像素。
//
// 模板参数 CHANNELS: 颜色通道数 (默认 3)
//
// 协作式加载:
//   所有线程通过共享内存 (__shared__) 协作加载当前瓦片的 Gaussian 数据，
//   每轮加载 BLOCK_SIZE=256 个 Gaussian，然后同步，再逐个处理。
//   这减少了全局内存访问次数，提高了带宽利用率。
//
// Alpha 合成 (论文公式 2, 3):
//   C = Σ c_i × α_i × T_i
//   T_{i+1} = T_i × (1 - α_i)
//   α_i = opacity_i × exp(-0.5 × Δ^T × Σ^{-1} × Δ)
//
// 提前终止优化:
//   1. power > 0 (像素在 Gaussian 1σ 之外很远) → skip
//   2. α < 1/255 (肉眼不可见) → skip
//   3. T × (1-α) < 0.0001 (后续贡献忽略) → done
// ============================================================================
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,          // 输入: 每个瓦片的 [start, end)
    const uint32_t* __restrict__ point_list,   // 输入: 排序后的 Gaussian ID 列表
    int W, int H,                              // 输入: 图像尺寸
    const float2* __restrict__ points_xy_image, // 输入: Gaussian 2D 屏幕坐标
    const float* __restrict__ features,        // 输入: Gaussian 颜色特征
    const float4* __restrict__ conic_opacity,  // 输入: {conic_x, conic_y, conic_z, opacity}
    float* __restrict__ final_T,               // 输出: 最终透射率
    uint32_t* __restrict__ n_contrib,          // 输出: 贡献的 Gaussian 数
    const float* __restrict__ bg_color,        // 输入: 背景颜色
    float* __restrict__ out_color,             // 输出: 渲染图像
    const float* __restrict__ depth,           // 输入: 每个 Gaussian 的深度
    float* __restrict__ out_depth)             // 输出: 深度图
{
    // ======================================================================
    // 初始化: 确定当前线程对应的瓦片和像素
    // ======================================================================

    // 获取 thread block 协作组句柄
    auto block = cg::this_thread_block();

    // 水平方向瓦片数量
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

    // pix_min: 当前瓦片左上角像素坐标
    uint2 pix_min = {
        block.group_index().x * BLOCK_X,
        block.group_index().y * BLOCK_Y
    };
    // pix_max: 当前瓦片右下角 (处理边界处不足一块的情况)
    uint2 pix_max = {
        min(pix_min.x + BLOCK_X, W),
        min(pix_min.y + BLOCK_Y, H)
    };
    // pix: 当前线程负责的像素
    uint2 pix = {
        pix_min.x + block.thread_index().x,
        pix_min.y + block.thread_index().y
    };
    // pix_id: 一维像素索引 (row-major)
    uint32_t pix_id = W * pix.y + pix.x;
    // pixf: 浮点数，用于计算像素到 Gaussian 中心的偏移
    float2 pixf = { (float)pix.x, (float)pix.y };

    // inside: 当前线程是否对应有效像素
    bool inside = pix.x < W && pix.y < H;
    // done: 边界外或已完成的像素标记
    // 边界外的线程仍参与协作加载但不写输出
    bool done = !inside;

    // ======================================================================
    // 加载当前瓦片的 Gaussian 范围
    // ======================================================================

    // range: 当前瓦片在排序后 point_list 中的 [start, end)
    uint2 range = ranges[block.group_index().y * horizontal_blocks
                         + block.group_index().x];

    // rounds: 需要多少轮批量加载
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // toDo: 剩余未处理 Gaussian 数
    int toDo = range.y - range.x;

    // ======================================================================
    // 共享内存: 协作加载 Gaussian 数据的缓冲区
    // ======================================================================
    // 每轮加载 BLOCK_SIZE 个 Gaussian → shared memory，
    // 所有线程同步后再逐个处理，避免重复全局内存访问。
    __shared__ int collected_id[BLOCK_SIZE];                // Gaussian ID
    __shared__ float2 collected_xy[BLOCK_SIZE];             // 屏幕坐标
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];  // 逆协方差 + 不透明度
    __shared__ float collected_depth[BLOCK_SIZE];           // 深度值

    // ======================================================================
    // 初始化累积变量
    // ======================================================================
    float T = 1.0f;              // 累积透射率: 从 1 开始 (完全透明)
    uint32_t contributor = 0;    // 已处理的 Gaussian 总数
    uint32_t last_contributor = 0;  // 最后一个有贡献的 Gaussian 序号
    float C[CHANNELS] = { 0 };   // 累积颜色 (每个通道)
    float D = 0.0f;              // 累积深度

    // ======================================================================
    // 外层循环: 分批次协作加载 Gaussian 数据
    // ======================================================================
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // 提前终止检查: block 内所有线程都 done 则退出
        // __syncthreads_count: 统计 block 内 done == true 的线程数
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        // 协作加载: 每个线程从全局内存加载一个 Gaussian
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            collected_depth[block.thread_rank()] = depth[coll_id];
        }
        block.sync();  // 等待所有线程完成加载

        // ==================================================================
        // 内层循环: 遍历当前 batch 的 Gaussian，逐像素 Alpha Blending
        // ==================================================================
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            contributor++;  // 记录共处理了多少 Gaussian

            // --- Step A: 像素到 Gaussian 中心的偏移 ---
            float2 xy = collected_xy[j];
            float2 d = { xy.x - pixf.x, xy.y - pixf.y };

            // --- Step B: 计算 2D Gaussian 在该像素处的指数值 ---
            // power = -0.5 × [dx, dy] × Σ^{-1} × [dx, dy]^T
            // Σ^{-1} = [conic.x  conic.y]
            //          [conic.y  conic.z]
            float4 con_o = collected_conic_opacity[j];
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y)
                          - con_o.y * d.x * d.y;

            if (power > 0.0f)
                continue;  // 1σ 范围外，跳过

            // --- Step C: 计算该像素处的 alpha ---
            // α = opacity × exp(power)
            // 论文 Eq.(2): α_i = o_i × G_i(x)
            // 钳位到 0.99 防数值不稳定 (论文附录)
            float alpha = min(0.99f, con_o.w * exp(power));

            // --- Step D: alpha 过小则跳过 ---
            if (alpha < 1.0f / 255.0f)
                continue;

            // --- Step E: 更新透射率并检查提前终止 ---
            float test_T = T * (1 - alpha);
            if (test_T < 0.0001f)
            {
                done = true;   // 后续 Gaussian 贡献 < 0.01%
                continue;
            }

            // --- Step F: Alpha Blending (论文 Eq.3) ---
            // C += c_i × α_i × T
            for (int ch = 0; ch < CHANNELS; ch++)
                C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

            // 深度累积: D += depth × α × T
            float dep = collected_depth[j];
            D += dep * alpha * T;

            // --- Step G: 更新透射率 ---
            T = test_T;
            last_contributor = contributor;
        }
    }

    // ======================================================================
    // 输出: 将最终结果写入全局内存
    // ======================================================================
    if (inside)
    {
        final_T[pix_id] = T;                   // 最终透射率
        n_contrib[pix_id] = last_contributor;  // 贡献的 Gaussian 数
        // 最终颜色 = 前景 + T × 背景色
        for (int ch = 0; ch < CHANNELS; ch++)
            out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
        out_depth[pix_id] = D;                 // 累积深度
    }
}

// ============================================================================
// FORWARD::render — Host 端渲染接口
// ============================================================================
// 负责 launch renderCUDA kernel。
// grid 维度 = (水平瓦片数, 垂直瓦片数)
// block 维度 = (BLOCK_X, BLOCK_Y) = (16, 16)
void FORWARD::render(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float2* means2D,
    const float* colors,
    const float4* conic_opacity,
    float* final_T,
    uint32_t* n_contrib,
    const float* bg_color,
    float* out_color,
    const float* depth,
    float* out_depth)
{
    renderCUDA<NUM_CHANNELS> <<<grid, block>>> (
        ranges,
        point_list,
        W, H,
        means2D,
        colors,
        conic_opacity,
        final_T,
        n_contrib,
        bg_color,
        out_color,
        depth,
        out_depth);
}

// ============================================================================
// FORWARD::preprocess — Host 端前处理接口
// ============================================================================
// 负责 launch preprocessCUDA kernel。
// 每个 Gaussian 一个线程，线程块大小固定为 256。
void FORWARD::preprocess(int P, int D, int M,
    const float* means3D,
    const glm::vec3* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const float* shs,
    bool* clamped,
    const float* cov3D_precomp,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    int* radii,
    float2* means2D,
    float* depths,
    float* cov3Ds,
    float* rgb,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered)
{
    // 网格大小: (P + 255) / 256, 刚好覆盖全部 P 个 Gaussian
    preprocessCUDA<NUM_CHANNELS> <<<(P + 255) / 256, 256>>> (
        P, D, M,
        means3D,
        scales,
        scale_modifier,
        rotations,
        opacities,
        shs,
        clamped,
        cov3D_precomp,
        colors_precomp,
        viewmatrix,
        projmatrix,
        cam_pos,
        W, H,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        radii,
        means2D,
        depths,
        cov3Ds,
        rgb,
        conic_opacity,
        grid,
        tiles_touched,
        prefiltered
    );
}

// ============================================================================
// 附录: 配置参数说明 (config.h)
// ============================================================================
// 以下常量定义在 cuda_rasterizer/config.h 中:
//
//   NUM_CHANNELS = 3      — 颜色通道数 (RGB)
//   BLOCK_X      = 16     — 瓦片/线程块宽度 (像素)
//   BLOCK_Y      = 16     — 瓦片/线程块高度 (像素)
//   BLOCK_SIZE   = 256    — BLOCK_X × BLOCK_Y
//
// 瓦片大小 16×16 的考量:
//   1. 每个 block 256 线程 = 8 个 warp，SM 占用率良好
//   2. 共享内存: 每个 batch 256 Gaussian × (4+8+16+4) = 8 KB
//      在 48 KB shared memory 预算内
//   3. 16×16 瓦片空间局部性好
//
// ============================================================================
// 附录: 辅助函数说明 (auxiliary.h)
// ============================================================================
//
// 1. ndc2Pix(v, S): NDC → 像素坐标
//    pixel = (v + 1) × S / 2 - 0.5
//
// 2. getRect(p, radius, rect_min, rect_max, grid):
//    像素 bounding box → 瓦片级别范围
//      rect_min = (p - radius) / BLOCK_X
//      rect_max = (p + radius + BLOCK_X - 1) / BLOCK_X
//
// 3. in_frustum(): 视锥体裁剪
//    检查: 深度 > 0.2, NDC 在 [-1.3, 1.3]
//
// 4. SH 系数常量:
//    SH_C0 = 0.28209... = 1/(2√π)
//    SH_C1 = 0.48860... = √(3/(4π))
//    SH_C2[5], SH_C3[7] — l=2, l=3 阶 Legendre 多项式系数
//
// ============================================================================

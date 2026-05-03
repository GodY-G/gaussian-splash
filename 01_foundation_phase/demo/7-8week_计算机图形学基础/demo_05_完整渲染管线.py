"""
Demo 05: 完整渲染管线（高分辨率 + 深度图 + Tile-based）
=========================================================
演示内容：
  1. 完整3DGS渲染管线（640×480）
  2. 深度图和不透明度图输出
  3. Tile-based渲染原理演示
  4. 渲染性能分析

与官方代码的对应：
  gaussian_renderer/__init__.py: render()
  submodules/diff-gaussian-rasterization/: CUDA光栅化器

运行：python demo_05_完整渲染管线.py
（注意：高分辨率渲染较慢，约需30-60秒）
"""

import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def rotation_y(theta_deg):
    theta = np.radians(theta_deg)
    return np.array([[ np.cos(theta), 0, np.sin(theta)],
                     [ 0,             1, 0            ],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rotation_x(theta_deg):
    theta = np.radians(theta_deg)
    return np.array([[1, 0,             0            ],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta),  np.cos(theta)]])

def project_gaussian(mu_3d, Sigma_3d, K, R_cam, t_cam):
    """3D高斯投影到2D"""
    fx, fy = K[0, 0], K[1, 1]
    mu_cam = R_cam @ mu_3d + t_cam
    x, y, z = mu_cam
    if z <= 0:
        return None, None, None

    J = np.array([[fx/z, 0, -fx*x/z**2],
                  [0, fy/z, -fy*y/z**2]])
    W = R_cam
    Sigma_2d = J @ W @ Sigma_3d @ W.T @ J.T

    # 添加低通滤波（防止高斯太小导致走样）
    Sigma_2d += np.eye(2) * 0.3

    mu_2d = np.array([fx*x/z + K[0,2], fy*y/z + K[1,2]])
    return mu_2d, Sigma_2d, z


def gaussian_2d_value(px, py, mu, Sigma_inv):
    """计算2D高斯在像素(px,py)处的值（向量化）"""
    dx = px - mu[0]
    dy = py - mu[1]
    exponent = -0.5 * (Sigma_inv[0,0]*dx*dx + 2*Sigma_inv[0,1]*dx*dy + Sigma_inv[1,1]*dy*dy)
    return np.exp(exponent)


# ─────────────────────────────────────────────
# 第一部分：完整渲染管线（含深度图和不透明度图）
# ─────────────────────────────────────────────

def render_full(gaussians, K, R_cam, t_cam, W, H, tile_size=16):
    """
    完整的3DGS渲染管线
    输出：颜色图、深度图、不透明度图、每像素高斯数量图

    对应官方代码：gaussian_renderer/__init__.py render()
    """
    # 步骤1：投影所有高斯
    projected = []
    for mu_3d, Sigma_3d, color, opacity in gaussians:
        mu_2d, Sigma_2d, depth = project_gaussian(mu_3d, Sigma_3d, K, R_cam, t_cam)
        if mu_2d is None:
            continue
        # 只保留在图像范围内的高斯（粗略剔除）
        if -50 < mu_2d[0] < W+50 and -50 < mu_2d[1] < H+50:
            Sigma_inv = np.linalg.inv(Sigma_2d)
            projected.append({
                'depth': depth,
                'mu_2d': mu_2d,
                'Sigma_inv': Sigma_inv,
                'color': np.array(color),
                'opacity': opacity,
            })

    # 步骤2：按深度排序（从近到远）
    projected.sort(key=lambda x: x['depth'])

    # 步骤3：Tile-based渲染
    # 将图像分成 tile_size×tile_size 的小块，每块独立处理
    color_map   = np.zeros((H, W, 3))
    depth_map   = np.zeros((H, W))
    opacity_map = np.zeros((H, W))
    count_map   = np.zeros((H, W), dtype=int)
    T_map       = np.ones((H, W))  # 透射率

    n_tiles_x = (W + tile_size - 1) // tile_size
    n_tiles_y = (H + tile_size - 1) // tile_size

    for g in projected:
        mu_2d = g['mu_2d']
        Sigma_inv = g['Sigma_inv']
        color = g['color']
        opacity = g['opacity']
        depth = g['depth']

        # 计算影响范围（3σ）
        # 用特征值估计最大半径
        eig_max = 1.0 / np.sqrt(min(Sigma_inv[0,0], Sigma_inv[1,1]) + 1e-6)
        radius = int(3 * eig_max) + 1

        u_min = max(0, int(mu_2d[0]) - radius)
        u_max = min(W, int(mu_2d[0]) + radius + 1)
        v_min = max(0, int(mu_2d[1]) - radius)
        v_max = min(H, int(mu_2d[1]) + radius + 1)

        if u_min >= u_max or v_min >= v_max:
            continue

        # 向量化计算该区域内所有像素的高斯值
        us = np.arange(u_min, u_max)
        vs = np.arange(v_min, v_max)
        UU, VV = np.meshgrid(us, vs)

        dx = UU - mu_2d[0]
        dy = VV - mu_2d[1]
        exponent = -0.5 * (Sigma_inv[0,0]*dx*dx + 2*Sigma_inv[0,1]*dx*dy + Sigma_inv[1,1]*dy*dy)
        g_val = np.exp(exponent)

        alpha = opacity * g_val  # (v_range, u_range)

        # Alpha合成
        T = T_map[v_min:v_max, u_min:u_max]
        contrib = T * alpha

        color_map[v_min:v_max, u_min:u_max] += contrib[:, :, np.newaxis] * color
        depth_map[v_min:v_max, u_min:u_max] += contrib * depth
        opacity_map[v_min:v_max, u_min:u_max] += contrib
        count_map[v_min:v_max, u_min:u_max] += (alpha > 0.01).astype(int)
        T_map[v_min:v_max, u_min:u_max] *= (1 - alpha)

    # 归一化深度图
    valid = opacity_map > 0.01
    depth_map[valid] /= opacity_map[valid]

    return (np.clip(color_map, 0, 1),
            depth_map,
            np.clip(opacity_map, 0, 1),
            count_map)


def make_building_scene(n_gaussians=500):
    """生成建筑场景的高斯基元"""
    np.random.seed(42)
    gaussians = []

    # 正面墙（大量扁平高斯）
    for _ in range(n_gaussians // 3):
        x = np.random.uniform(-3, 3)
        y = np.random.uniform(-2, 2)
        z = np.random.uniform(-0.1, 0.1)
        mu = np.array([x, y, z])
        # 扁平高斯（墙面）
        R = rotation_y(np.random.uniform(-5, 5))
        S = np.diag([np.random.uniform(0.2, 0.5),
                     np.random.uniform(0.2, 0.5),
                     np.random.uniform(0.02, 0.05)])
        Sigma = R @ S @ S.T @ R.T
        # 墙面颜色（米白色，略有变化）
        color = np.clip([0.85 + np.random.randn()*0.05,
                         0.82 + np.random.randn()*0.05,
                         0.75 + np.random.randn()*0.05], 0, 1)
        gaussians.append((mu, Sigma, color, np.random.uniform(0.7, 0.9)))

    # 窗户（深色，小高斯）
    for wx in [-2, -1, 0, 1, 2]:
        for wy in [-1, 0, 1]:
            mu = np.array([wx + np.random.randn()*0.05,
                           wy + np.random.randn()*0.05,
                           0.02])
            S = np.diag([0.25, 0.35, 0.01])
            Sigma = S @ S.T
            color = np.clip([0.2 + np.random.randn()*0.05,
                             0.3 + np.random.randn()*0.05,
                             0.5 + np.random.randn()*0.05], 0, 1)
            gaussians.append((mu, Sigma, color, 0.95))

    # 地面（水平扁平高斯）
    for _ in range(n_gaussians // 4):
        x = np.random.uniform(-4, 4)
        z = np.random.uniform(-1, 3)
        mu = np.array([x, -2.2, z])
        S = np.diag([np.random.uniform(0.3, 0.8),
                     np.random.uniform(0.02, 0.05),
                     np.random.uniform(0.3, 0.8)])
        Sigma = S @ S.T
        color = np.clip([0.4 + np.random.randn()*0.05,
                         0.4 + np.random.randn()*0.05,
                         0.35 + np.random.randn()*0.05], 0, 1)
        gaussians.append((mu, Sigma, color, np.random.uniform(0.6, 0.8)))

    # 天空（远处大高斯）
    for _ in range(50):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(2, 5)
        mu = np.array([x, y, 8])
        S = np.diag([1.5, 1.5, 0.5])
        Sigma = S @ S.T
        color = np.clip([0.5 + np.random.randn()*0.05,
                         0.7 + np.random.randn()*0.05,
                         0.95 + np.random.randn()*0.03], 0, 1)
        gaussians.append((mu, Sigma, color, 0.3))

    return gaussians


def demo_full_render():
    """完整渲染管线演示"""
    print("="*60)
    print("完整3DGS渲染管线（640×480）")
    print("="*60)

    W, H = 640, 480
    K = np.array([[500, 0, W/2], [0, 500, H/2], [0, 0, 1]], dtype=float)
    R_cam = rotation_x(-5)
    t_cam = np.array([0, 0.5, 7])

    gaussians = make_building_scene(n_gaussians=500)
    print(f"高斯基元数量: {len(gaussians)}")

    print("渲染中...")
    t0 = time.time()
    color_map, depth_map, opacity_map, count_map = render_full(
        gaussians, K, R_cam, t_cam, W, H)
    t1 = time.time()
    print(f"渲染耗时: {t1-t0:.2f}秒（Python实现，官方CUDA版本快1000倍以上）")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"完整3DGS渲染管线输出（{W}×{H}，{len(gaussians)}个高斯基元）",
                 fontsize=12, fontweight='bold')

    axes[0, 0].imshow(color_map)
    axes[0, 0].set_title('颜色图（RGB）')
    axes[0, 0].axis('off')

    # 深度图（归一化显示）
    depth_vis = depth_map.copy()
    depth_vis[depth_vis == 0] = depth_vis[depth_vis > 0].max() if np.any(depth_vis > 0) else 1
    im_depth = axes[0, 1].imshow(depth_vis, cmap='plasma_r')
    axes[0, 1].set_title('深度图（近=亮，远=暗）')
    axes[0, 1].axis('off')
    plt.colorbar(im_depth, ax=axes[0, 1], fraction=0.046)

    im_opacity = axes[1, 0].imshow(opacity_map, cmap='gray')
    axes[1, 0].set_title('不透明度图（白=不透明，黑=透明）')
    axes[1, 0].axis('off')
    plt.colorbar(im_opacity, ax=axes[1, 0], fraction=0.046)

    im_count = axes[1, 1].imshow(count_map, cmap='hot')
    axes[1, 1].set_title('每像素高斯数量\n（热图：红=多，黑=少）')
    axes[1, 1].axis('off')
    plt.colorbar(im_count, ax=axes[1, 1], fraction=0.046)

    plt.tight_layout()
    plt.savefig('output_05a_完整渲染.png', dpi=120, bbox_inches='tight')
    print("已保存: output_05a_完整渲染.png")
    plt.show()

    print(f"\n渲染统计：")
    print(f"  平均深度: {depth_map[depth_map>0].mean():.2f}")
    print(f"  平均不透明度: {opacity_map.mean():.3f}")
    print(f"  平均每像素高斯数: {count_map.mean():.1f}")
    print(f"  最大每像素高斯数: {count_map.max()}")


# ─────────────────────────────────────────────
# 第二部分：Tile-based渲染原理
# ─────────────────────────────────────────────

def demo_tile_based_rendering():
    """
    Tile-based渲染原理演示
    3DGS官方实现将图像分成16×16的tile，
    每个tile独立处理，便于GPU并行化

    对应官方代码：submodules/diff-gaussian-rasterization/
    """
    print("\n" + "="*60)
    print("Tile-based渲染原理")
    print("="*60)

    W, H = 320, 240
    tile_size = 16
    n_tiles_x = W // tile_size
    n_tiles_y = H // tile_size

    # 生成一些高斯基元
    np.random.seed(42)
    n_gaussians = 50
    gaussians_2d = []
    for _ in range(n_gaussians):
        mu = np.array([np.random.uniform(0, W), np.random.uniform(0, H)])
        sigma = np.random.uniform(10, 40)
        color = np.random.rand(3)
        gaussians_2d.append((mu, sigma, color))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Tile-based渲染：将图像分块，每块独立处理（便于GPU并行）",
                 fontsize=12, fontweight='bold')

    # 左图：显示tile划分和高斯分布
    ax = axes[0]
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(f'图像分成 {n_tiles_x}×{n_tiles_y} 个Tile\n每个Tile={tile_size}×{tile_size}像素')

    # 画tile网格
    for i in range(0, W+1, tile_size):
        ax.axvline(i, color='gray', alpha=0.3, linewidth=0.5)
    for j in range(0, H+1, tile_size):
        ax.axhline(j, color='gray', alpha=0.3, linewidth=0.5)

    # 画高斯基元
    for mu, sigma, color in gaussians_2d:
        circle = plt.Circle(mu, sigma, color=color, alpha=0.3)
        ax.add_patch(circle)
        ax.plot(*mu, 'o', color=color, markersize=3)

    # 高亮一个tile，显示它包含哪些高斯
    highlight_tile = (3, 2)  # (tile_x, tile_y)
    tx, ty = highlight_tile
    tile_rect = plt.Rectangle((tx*tile_size, ty*tile_size), tile_size, tile_size,
                               edgecolor='red', facecolor='red', alpha=0.2, linewidth=2)
    ax.add_patch(tile_rect)
    ax.text(tx*tile_size + tile_size/2, ty*tile_size + tile_size/2,
            f'Tile\n({tx},{ty})', ha='center', va='center', color='red', fontsize=8)

    ax.set_xlabel('u（像素）')
    ax.set_ylabel('v（像素）')

    # 右图：统计每个tile包含的高斯数量
    ax2 = axes[1]
    tile_counts = np.zeros((n_tiles_y, n_tiles_x))

    for mu, sigma, _ in gaussians_2d:
        # 找到这个高斯影响的tile范围
        t_x_min = max(0, int((mu[0] - 3*sigma) / tile_size))
        t_x_max = min(n_tiles_x-1, int((mu[0] + 3*sigma) / tile_size))
        t_y_min = max(0, int((mu[1] - 3*sigma) / tile_size))
        t_y_max = min(n_tiles_y-1, int((mu[1] + 3*sigma) / tile_size))
        tile_counts[t_y_min:t_y_max+1, t_x_min:t_x_max+1] += 1

    im = ax2.imshow(tile_counts, cmap='hot', aspect='auto')
    ax2.set_title(f'每个Tile包含的高斯数量\n（热图：红=多，黑=少）')
    ax2.set_xlabel('Tile X')
    ax2.set_ylabel('Tile Y')
    plt.colorbar(im, ax=ax2, fraction=0.046)

    print(f"Tile统计：")
    print(f"  总Tile数: {n_tiles_x * n_tiles_y}")
    print(f"  平均每Tile高斯数: {tile_counts.mean():.1f}")
    print(f"  最大每Tile高斯数: {tile_counts.max():.0f}")
    print(f"\n官方代码中的Tile处理：")
    print(f"  1. 对每个高斯，计算它覆盖的Tile范围")
    print(f"  2. 为每个(Tile, 高斯)对创建一个排序键（Tile_ID + 深度）")
    print(f"  3. 按排序键排序（GPU并行基数排序）")
    print(f"  4. 每个Tile独立执行alpha合成（GPU并行）")

    plt.tight_layout()
    plt.savefig('output_05b_Tile渲染.png', dpi=120, bbox_inches='tight')
    print("\n已保存: output_05b_Tile渲染.png")
    plt.show()


# ─────────────────────────────────────────────
# 第三部分：多视角渲染
# ─────────────────────────────────────────────

def demo_multi_view_render():
    """从不同视角渲染同一场景"""
    print("\n" + "="*60)
    print("多视角渲染（模拟3DGS训练时的输入）")
    print("="*60)

    W, H = 320, 240
    K = np.array([[300, 0, W/2], [0, 300, H/2], [0, 0, 1]], dtype=float)
    gaussians = make_building_scene(n_gaussians=300)

    angles = [-30, -15, 0, 15, 30]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("多视角渲染（3DGS训练时从多个视角监督）", fontsize=12, fontweight='bold')

    for ax, angle in zip(axes, angles):
        R_cam = rotation_x(-5) @ rotation_y(angle)
        t_cam = np.array([0, 0.5, 7])

        color_map, _, _, _ = render_full(gaussians, K, R_cam, t_cam, W, H)
        ax.imshow(color_map)
        ax.set_title(f'视角 {angle:+d}°', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('output_05c_多视角渲染.png', dpi=120, bbox_inches='tight')
    print("已保存: output_05c_多视角渲染.png")
    plt.show()


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Demo 05: 完整渲染管线")
    print("=" * 60)

    demo_full_render()
    demo_tile_based_rendering()
    demo_multi_view_render()

    print("\n全部演示完成！")
    print("\n与官方代码的对应：")
    print("  gaussian_renderer/__init__.py: render() — 前向渲染")
    print("  submodules/diff-gaussian-rasterization/: CUDA光栅化器")
    print("  官方实现用CUDA并行化，速度比本Python实现快1000倍以上")
    print("  本demo的目的是理解原理，不是追求速度")

"""
Demo 02: 坐标变换
=================
演示内容：
  1. 平移、旋转、缩放矩阵（齐次坐标）
  2. 变换复合：顺序很重要
  3. 3DGS投影链：世界坐标 → 相机坐标 → 图像坐标

运行：python demo_02_坐标变换.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# 齐次坐标变换矩阵工具函数
# ─────────────────────────────────────────────

def translation_2d(tx, ty):
    """2D平移矩阵（3×3齐次）"""
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1 ]], dtype=float)

def rotation_2d(theta_deg):
    """2D旋转矩阵（3×3齐次）"""
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)

def scale_2d(sx, sy):
    """2D缩放矩阵（3×3齐次）"""
    return np.array([[sx, 0,  0],
                     [0,  sy, 0],
                     [0,  0,  1]], dtype=float)

def apply_transform(M, points):
    """对点集应用变换矩阵（points: 2×N）"""
    ones = np.ones((1, points.shape[1]))
    pts_h = np.vstack([points, ones])   # 3×N 齐次坐标
    transformed = M @ pts_h
    return transformed[:2]              # 返回2×N


# ─────────────────────────────────────────────
# 第一部分：三种基本变换
# ─────────────────────────────────────────────

def make_arrow_shape():
    """创建一个箭头形状的点集，用于演示变换"""
    pts = np.array([
        [0, 0], [2, 0], [2, -0.5], [3, 1], [2, 2.5], [2, 2], [0, 2]
    ], dtype=float).T
    return pts

def demo_basic_transforms():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("三种基本变换（齐次坐标）", fontsize=14, fontweight='bold')

    shape = make_arrow_shape()

    configs = [
        (translation_2d(2, 1),  "平移 (tx=2, ty=1)",  "blue"),
        (rotation_2d(45),        "旋转 45°",            "green"),
        (scale_2d(1.5, 0.7),    "缩放 (sx=1.5, sy=0.7)", "orange"),
    ]

    for ax, (M, title, color) in zip(axes, configs):
        ax.set_xlim(-1, 6)
        ax.set_ylim(-2, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_title(title, fontsize=11)

        # 原始形状（灰色虚线）
        orig = np.hstack([shape, shape[:, :1]])
        ax.fill(orig[0], orig[1], alpha=0.15, color='gray')
        ax.plot(orig[0], orig[1], 'gray', linestyle='--', linewidth=1.5, label='原始')

        # 变换后形状
        new_pts = apply_transform(M, shape)
        new_pts_closed = np.hstack([new_pts, new_pts[:, :1]])
        ax.fill(new_pts_closed[0], new_pts_closed[1], alpha=0.4, color=color)
        ax.plot(new_pts_closed[0], new_pts_closed[1], color=color, linewidth=2, label='变换后')

        ax.legend(fontsize=9)

        # 显示矩阵
        mat_str = '\n'.join([str(np.round(row, 2).tolist()) for row in M])
        ax.text(0.02, 0.02, f'M =\n{mat_str}', transform=ax.transAxes,
                fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig('output_02a_基本变换.png', dpi=120, bbox_inches='tight')
    print("已保存: output_02a_基本变换.png")
    plt.show()


# ─────────────────────────────────────────────
# 第二部分：变换复合——顺序很重要！
# ─────────────────────────────────────────────

def demo_transform_order():
    """演示先旋转后平移 vs 先平移后旋转的区别"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("变换顺序很重要：先旋转后平移 ≠ 先平移后旋转", fontsize=13, fontweight='bold')

    shape = make_arrow_shape()
    T = translation_2d(3, 0)
    R = rotation_2d(90)

    # 先旋转后平移：M = T · R（先应用R，再应用T）
    M1 = T @ R
    # 先平移后旋转：M = R · T
    M2 = R @ T

    for ax, (M, title, color) in zip(axes, [
        (M1, "先旋转90°，再平移(3,0)\nM = T · R", "blue"),
        (M2, "先平移(3,0)，再旋转90°\nM = R · T", "red"),
    ]):
        ax.set_xlim(-5, 7)
        ax.set_ylim(-5, 7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_title(title, fontsize=11)

        # 原始
        orig = np.hstack([shape, shape[:, :1]])
        ax.fill(orig[0], orig[1], alpha=0.15, color='gray')
        ax.plot(orig[0], orig[1], 'gray', linestyle='--', linewidth=1.5, label='原始')

        # 变换后
        new_pts = apply_transform(M, shape)
        new_pts_closed = np.hstack([new_pts, new_pts[:, :1]])
        ax.fill(new_pts_closed[0], new_pts_closed[1], alpha=0.4, color=color)
        ax.plot(new_pts_closed[0], new_pts_closed[1], color=color, linewidth=2, label='变换后')

        # 标注旋转中心
        ax.plot(0, 0, 'ko', markersize=8)
        ax.text(0.1, 0.1, '旋转中心(0,0)', fontsize=9)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('output_02b_变换顺序.png', dpi=120, bbox_inches='tight')
    print("已保存: output_02b_变换顺序.png")
    plt.show()


# ─────────────────────────────────────────────
# 第三部分：3DGS投影链
# ─────────────────────────────────────────────

def demo_3dgs_projection():
    """
    模拟3DGS的完整投影链：
    世界坐标 → 相机坐标 → 图像坐标

    p_image = K · [R|t] · P_world
    """
    print("\n" + "="*60)
    print("3DGS 投影链演示：世界坐标 → 相机坐标 → 图像坐标")
    print("="*60)

    # ── 相机参数设置 ──
    # 内参矩阵 K（焦距 fx=fy=500，主点在图像中心 cx=320, cy=240）
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0
    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]])

    # 外参：相机在世界坐标 (0, 0, 5) 处，朝向 -Z 方向（看向原点）
    # 旋转：绕Y轴旋转30度
    theta = np.radians(30)
    R = np.array([[ np.cos(theta), 0, np.sin(theta)],
                  [ 0,             1, 0            ],
                  [-np.sin(theta), 0, np.cos(theta)]])
    t = np.array([[0], [0], [5]])  # 相机位置（世界坐标）

    print(f"\n内参矩阵 K:\n{K}")
    print(f"\n旋转矩阵 R（绕Y轴30°）:\n{np.round(R, 4)}")
    print(f"\n平移向量 t（相机在世界坐标中的位置）:\n{t.T}")

    # ── 世界坐标中的3D点 ──
    # 模拟一个简单的建筑轮廓（正方形）
    world_points = np.array([
        [-1, -1, 0],
        [ 1, -1, 0],
        [ 1,  1, 0],
        [-1,  1, 0],
        [ 0,  0, 1],  # 顶点
    ], dtype=float).T  # 3×N

    print(f"\n世界坐标中的3D点（{world_points.shape[1]}个）:")
    for i, p in enumerate(world_points.T):
        print(f"  P{i} = {p}")

    # ── 步骤1：世界坐标 → 相机坐标 ──
    # P_cam = R · P_world + t
    cam_points = R @ world_points + t
    print(f"\n步骤1：相机坐标 P_cam = R·P + t")
    for i, p in enumerate(cam_points.T):
        print(f"  P{i}_cam = {np.round(p, 3)}")

    # ── 步骤2：相机坐标 → 图像坐标（透视投影）──
    # p_image = K · P_cam，然后除以深度 z
    img_points_h = K @ cam_points          # 3×N（齐次）
    img_points = img_points_h[:2] / img_points_h[2]  # 2×N（除以深度）

    print(f"\n步骤2：图像坐标 p = K·P_cam / z")
    for i, p in enumerate(img_points.T):
        print(f"  p{i}_image = ({p[0]:.1f}, {p[1]:.1f}) 像素")

    # ── 可视化 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("3DGS 投影链：p = K · [R|t] · P", fontsize=13, fontweight='bold')

    # 左图：3D世界坐标
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.scatter(*world_points, c='red', s=80, zorder=5)
    for i, p in enumerate(world_points.T):
        ax3d.text(p[0]+0.05, p[1]+0.05, p[2]+0.05, f'P{i}', fontsize=9)

    # 画相机位置
    cam_pos_world = -R.T @ t  # 相机在世界坐标中的位置
    ax3d.scatter(*cam_pos_world, c='blue', s=150, marker='^', zorder=5)
    ax3d.text(cam_pos_world[0, 0]+0.1, cam_pos_world[1, 0], cam_pos_world[2, 0], '相机', color='blue', fontsize=10)

    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('世界坐标系（3D）')

    # 右图：图像坐标（投影结果）
    ax2d = axes[1]
    ax2d.set_xlim(0, 640)
    ax2d.set_ylim(480, 0)  # 图像坐标Y轴向下
    ax2d.set_aspect('equal')
    ax2d.grid(True, alpha=0.3)
    ax2d.set_title('图像坐标系（2D投影结果）')
    ax2d.set_xlabel('u（像素）')
    ax2d.set_ylabel('v（像素）')

    # 画投影点
    ax2d.scatter(img_points[0], img_points[1], c='red', s=80, zorder=5)
    for i, p in enumerate(img_points.T):
        ax2d.text(p[0]+5, p[1]+5, f'p{i}({p[0]:.0f},{p[1]:.0f})', fontsize=9)

    # 连线（模拟建筑轮廓）
    order = [0, 1, 2, 3, 0]
    ax2d.plot(img_points[0, order], img_points[1, order], 'b-', linewidth=2, label='投影轮廓')
    for i in range(4):
        ax2d.plot([img_points[0, i], img_points[0, 4]],
                  [img_points[1, i], img_points[1, 4]], 'g--', alpha=0.5)

    # 标注主点
    ax2d.plot(cx, cy, 'b+', markersize=15, markeredgewidth=2)
    ax2d.text(cx+5, cy+10, f'主点({cx:.0f},{cy:.0f})', color='blue', fontsize=9)
    ax2d.legend()

    plt.tight_layout()
    plt.savefig('output_02c_投影链.png', dpi=120, bbox_inches='tight')
    print("\n已保存: output_02c_投影链.png")
    plt.show()


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Demo 02: 坐标变换")
    print("=" * 50)

    demo_basic_transforms()
    demo_transform_order()
    demo_3dgs_projection()

    print("\n全部演示完成！")

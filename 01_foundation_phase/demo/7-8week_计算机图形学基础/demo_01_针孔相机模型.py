"""
Demo 01: 针孔相机模型
=====================
演示内容：
  1. 针孔相机投影：3D点 → 2D图像
  2. 内参矩阵K的作用（焦距、主点）
  3. 外参矩阵[R|t]的作用（相机位姿）
  4. 多相机视角可视化

运行：python demo_01_针孔相机模型.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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

def project_points(points_3d, K, R, t):
    """
    将3D点投影到图像平面
    points_3d: N×3
    返回: N×2 图像坐标
    """
    # 世界坐标 → 相机坐标
    P_cam = (R @ points_3d.T + t.reshape(3, 1)).T  # N×3

    # 过滤掉在相机后面的点
    valid = P_cam[:, 2] > 0

    # 相机坐标 → 图像坐标
    p_img = np.zeros((len(points_3d), 2))
    p_img[valid, 0] = K[0, 0] * P_cam[valid, 0] / P_cam[valid, 2] + K[0, 2]
    p_img[valid, 1] = K[1, 1] * P_cam[valid, 1] / P_cam[valid, 2] + K[1, 2]
    p_img[~valid] = np.nan

    return p_img, valid


def make_building_3d():
    """创建一个简单的建筑3D模型（线框）"""
    # 建筑主体（长方体）
    w, h, d = 4, 3, 2  # 宽、高、深
    vertices = np.array([
        # 前面
        [-w/2, -h/2, 0], [w/2, -h/2, 0], [w/2, h/2, 0], [-w/2, h/2, 0],
        # 后面
        [-w/2, -h/2, d], [w/2, -h/2, d], [w/2, h/2, d], [-w/2, h/2, d],
        # 屋顶三角形
        [0, h/2+1.5, d/2],
    ])

    edges = [
        (0,1),(1,2),(2,3),(3,0),  # 前面
        (4,5),(5,6),(6,7),(7,4),  # 后面
        (0,4),(1,5),(2,6),(3,7),  # 侧边
        (3,8),(2,8),(6,8),(7,8),  # 屋顶
    ]
    return vertices, edges


def demo_camera_projection():
    """演示相机投影的完整过程"""
    vertices, edges = make_building_3d()

    # 相机参数
    W, H = 640, 480
    fx = fy = 500.0
    cx, cy = W/2, H/2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # 相机位姿：在建筑正前方，稍微偏上
    R = rotation_x(-10) @ rotation_y(0)
    t = np.array([0, 0.5, 8])  # 相机在世界坐标 (0, -0.5, -8) 处

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("针孔相机模型：3D建筑 → 2D图像投影", fontsize=13, fontweight='bold')

    # 左图：3D场景
    ax3d = fig.add_subplot(121, projection='3d')
    for i, j in edges:
        ax3d.plot([vertices[i,0], vertices[j,0]],
                  [vertices[i,1], vertices[j,1]],
                  [vertices[i,2], vertices[j,2]], 'b-', linewidth=1.5)
    ax3d.scatter(vertices[:,0], vertices[:,1], vertices[:,2], c='blue', s=30)

    # 画相机位置
    cam_pos = -R.T @ t
    ax3d.scatter(*cam_pos, c='red', s=200, marker='^', zorder=5)
    ax3d.text(cam_pos[0], cam_pos[1], cam_pos[2]+0.3, '相机', color='red', fontsize=10)

    # 画光线（从相机到几个顶点）
    for v in vertices[:4]:
        ax3d.plot([cam_pos[0], v[0]], [cam_pos[1], v[1]], [cam_pos[2], v[2]],
                  'r--', alpha=0.3, linewidth=0.8)

    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('3D场景（世界坐标系）')

    # 右图：投影结果
    ax2d = axes[1]
    ax2d.set_xlim(0, W)
    ax2d.set_ylim(H, 0)  # 图像坐标Y轴向下
    ax2d.set_facecolor('black')
    ax2d.set_title(f'相机图像（{W}×{H}像素）\nfx=fy={fx:.0f}, cx={cx:.0f}, cy={cy:.0f}')

    proj, valid = project_points(vertices, K, R, t)

    for i, j in edges:
        if valid[i] and valid[j]:
            ax2d.plot([proj[i,0], proj[j,0]], [proj[i,1], proj[j,1]],
                      'cyan', linewidth=2)

    ax2d.scatter(proj[valid,0], proj[valid,1], c='yellow', s=50, zorder=5)
    ax2d.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2, label=f'主点({cx:.0f},{cy:.0f})')
    ax2d.legend(fontsize=9)
    ax2d.set_xlabel('u（像素）')
    ax2d.set_ylabel('v（像素）')
    ax2d.grid(True, alpha=0.2, color='gray')

    plt.tight_layout()
    plt.savefig('output_01a_相机投影.png', dpi=120, bbox_inches='tight')
    print("已保存: output_01a_相机投影.png")
    plt.show()


def demo_focal_length_effect():
    """焦距对投影的影响"""
    vertices, edges = make_building_3d()
    W, H = 640, 480
    cx, cy = W/2, H/2
    R = rotation_x(-10)
    t = np.array([0, 0.5, 8])

    focal_lengths = [200, 400, 600, 1000]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("焦距对投影的影响（焦距越大，视角越窄，物体越大）", fontsize=12, fontweight='bold')

    for ax, f in zip(axes, focal_lengths):
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        proj, valid = project_points(vertices, K, R, t)

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_facecolor('black')
        ax.set_title(f'焦距 f={f}px\n视角≈{2*np.degrees(np.arctan(W/(2*f))):.0f}°', fontsize=10)

        for i, j in edges:
            if valid[i] and valid[j]:
                ax.plot([proj[i,0], proj[j,0]], [proj[i,1], proj[j,1]],
                        'cyan', linewidth=1.5)
        ax.scatter(proj[valid,0], proj[valid,1], c='yellow', s=20)
        ax.set_xlabel('u')
        ax.set_ylabel('v')

    plt.tight_layout()
    plt.savefig('output_01b_焦距影响.png', dpi=120, bbox_inches='tight')
    print("已保存: output_01b_焦距影响.png")
    plt.show()


def demo_multi_view():
    """多视角投影（模拟COLMAP的多视角输入）"""
    vertices, edges = make_building_3d()
    W, H = 320, 240
    K = np.array([[400, 0, W/2], [0, 400, H/2], [0, 0, 1]])

    # 围绕建筑的多个相机位置
    angles = np.linspace(-60, 60, 5)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("多视角投影（COLMAP输入的多张照片）", fontsize=12, fontweight='bold')

    for ax, angle in zip(axes, angles):
        R = rotation_y(angle)
        # 相机始终朝向建筑中心
        cam_dist = 8
        cam_x = cam_dist * np.sin(np.radians(angle))
        cam_z = cam_dist * np.cos(np.radians(angle))
        t = R @ np.array([0, 0.5, cam_dist])

        proj, valid = project_points(vertices, K, R, t)

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_facecolor('#1a1a2e')
        ax.set_title(f'视角 {angle:+.0f}°', fontsize=10)

        for i, j in edges:
            if valid[i] and valid[j]:
                ax.plot([proj[i,0], proj[j,0]], [proj[i,1], proj[j,1]],
                        'lightblue', linewidth=1.5)
        ax.scatter(proj[valid,0], proj[valid,1], c='yellow', s=15)
        ax.set_xlabel('u')
        ax.set_ylabel('v')

    plt.tight_layout()
    plt.savefig('output_01c_多视角.png', dpi=120, bbox_inches='tight')
    print("已保存: output_01c_多视角.png")
    plt.show()


if __name__ == '__main__':
    print("Demo 01: 针孔相机模型")
    print("=" * 50)
    demo_camera_projection()
    demo_focal_length_effect()
    demo_multi_view()
    print("\n全部演示完成！")

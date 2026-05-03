"""
Demo 05: 四元数与旋转 —— 3DGS参数化完整链路
=============================================
演示内容：
  1. 四元数的几何意义（绕轴旋转）
  2. 四元数 → 旋转矩阵（与官方代码一致）
  3. 完整链路：四元数 + 缩放 → 协方差矩阵 → 2D投影
  4. 对比：直接优化Σ vs 优化(q,s)的数值稳定性

与官方代码的对应：
  utils/general_utils.py: build_rotation()
  scene/gaussian_model.py: get_covariance()

运行：python demo_05_四元数与旋转.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# 四元数工具函数（与官方代码逻辑一致）
# ─────────────────────────────────────────────

def quaternion_to_rotation_matrix(q):
    """
    四元数 → 旋转矩阵
    q = (w, x, y, z)，单位四元数

    对应官方代码：utils/general_utils.py build_rotation()
    """
    # 归一化（确保单位四元数）
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    R = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),   1 - 2*(x*x + y*y)],
    ])
    return R


def axis_angle_to_quaternion(axis, angle_deg):
    """
    轴角表示 → 四元数
    axis: 旋转轴（单位向量）
    angle_deg: 旋转角度（度）
    """
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(angle_deg)
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def build_covariance_3d(q, s):
    """
    从四元数q和缩放s构建3D协方差矩阵
    Σ = R · diag(s)² · Rᵀ

    对应官方代码：scene/gaussian_model.py get_covariance()
    """
    R = quaternion_to_rotation_matrix(q)
    S = np.diag(s)
    return R @ S @ S.T @ R.T


# ─────────────────────────────────────────────
# 第一部分：四元数的几何意义
# ─────────────────────────────────────────────

def demo_quaternion_geometry():
    """可视化四元数表示的旋转"""
    print("="*60)
    print("四元数的几何意义：绕任意轴旋转")
    print("="*60)

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("四元数 q=(w,x,y,z) 表示绕轴旋转\nq = (cos(θ/2), sin(θ/2)·axis)",
                 fontsize=12, fontweight='bold')

    # 三种旋转
    rotations = [
        (np.array([0, 0, 1]), 45,  "绕Z轴旋转45°\nq=(cos22.5°, 0, 0, sin22.5°)"),
        (np.array([1, 0, 0]), 90,  "绕X轴旋转90°\nq=(cos45°, sin45°, 0, 0)"),
        (np.array([1, 1, 0])/np.sqrt(2), 60, "绕(1,1,0)轴旋转60°"),
    ]

    # 原始向量（一个箭头）
    v_orig = np.array([1, 0, 0])

    for idx, (axis, angle_deg, title) in enumerate(rotations):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')

        q = axis_angle_to_quaternion(axis, angle_deg)
        R = quaternion_to_rotation_matrix(q)
        v_rot = R @ v_orig

        print(f"\n{title.split(chr(10))[0]}:")
        print(f"  四元数 q = ({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f})")
        print(f"  旋转矩阵 R =\n{np.round(R, 4)}")
        print(f"  验证 det(R) = {np.linalg.det(R):.6f}（应为1）")
        print(f"  验证 RᵀR = I: {np.allclose(R.T @ R, np.eye(3))}")

        # 画坐标轴
        for vec, color, label in [
            ([1,0,0], 'red',   'X'),
            ([0,1,0], 'green', 'Y'),
            ([0,0,1], 'blue',  'Z'),
        ]:
            ax.quiver(0,0,0, *vec, color=color, alpha=0.3, arrow_length_ratio=0.1)

        # 画旋转轴
        ax.quiver(0,0,0, *axis*1.5, color='purple', linewidth=3,
                  arrow_length_ratio=0.1, label='旋转轴')

        # 画原始向量和旋转后向量
        ax.quiver(0,0,0, *v_orig, color='orange', linewidth=3,
                  arrow_length_ratio=0.15, label='原始')
        ax.quiver(0,0,0, *v_rot, color='cyan', linewidth=3,
                  arrow_length_ratio=0.15, label='旋转后')

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    plt.savefig('output_05a_四元数旋转.png', dpi=120, bbox_inches='tight')
    print("\n已保存: output_05a_四元数旋转.png")
    plt.show()


# ─────────────────────────────────────────────
# 第二部分：完整链路 q + s → Σ → Σ'
# ─────────────────────────────────────────────

def project_covariance_2d(Sigma_3d, mu_cam, fx, fy):
    """
    3D协方差矩阵投影到2D（雅可比近似）
    Σ' = J · W · Σ · Wᵀ · Jᵀ

    对应官方代码：gaussian_renderer/__init__.py 中的投影部分
    """
    x, y, z = mu_cam
    J = np.array([
        [fx/z,    0,   -fx*x/z**2],
        [0,    fy/z,   -fy*y/z**2],
    ])
    # W = 相机旋转矩阵（这里简化为单位矩阵，即相机坐标系=世界坐标系）
    W = np.eye(3)
    return J @ W @ Sigma_3d @ W.T @ J.T


def demo_full_pipeline():
    """演示完整的参数化链路"""
    print("\n" + "="*60)
    print("完整链路：四元数q + 缩放s → Σ → Σ'（2D投影）")
    print("="*60)

    fx, fy = 500.0, 500.0
    mu_cam = np.array([0.5, 0.3, 5.0])  # 高斯在相机坐标系中的位置

    # 不同的旋转和缩放组合
    configs = [
        (axis_angle_to_quaternion([0,0,1], 0),   [1.5, 0.5, 0.8], "无旋转\ns=(1.5,0.5,0.8)"),
        (axis_angle_to_quaternion([0,0,1], 45),  [1.5, 0.5, 0.8], "绕Z轴45°\ns=(1.5,0.5,0.8)"),
        (axis_angle_to_quaternion([1,0,0], 30),  [2.0, 0.3, 0.3], "绕X轴30°\ns=(2.0,0.3,0.3)"),
        (axis_angle_to_quaternion([0,1,0], 60),  [1.0, 1.0, 2.0], "绕Y轴60°\ns=(1.0,1.0,2.0)"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("完整链路：四元数q + 缩放s → 3D协方差Σ → 2D投影Σ'",
                 fontsize=12, fontweight='bold')

    for col, (q, s, title) in enumerate(configs):
        s = np.array(s)
        R = quaternion_to_rotation_matrix(q)
        Sigma_3d = build_covariance_3d(q, s)
        Sigma_2d = project_covariance_2d(Sigma_3d, mu_cam, fx, fy)

        print(f"\n{title.split(chr(10))[0]}:")
        print(f"  q = {np.round(q, 4)}")
        print(f"  s = {s}")
        print(f"  Σ_3d 特征值 = {np.round(np.linalg.eigvalsh(Sigma_3d), 4)}")
        print(f"  Σ_2d 特征值 = {np.round(np.linalg.eigvalsh(Sigma_2d), 4)}")

        # 上行：3D椭球体（XY截面）
        ax3d = axes[0, col]
        theta = np.linspace(0, 2*np.pi, 100)
        # 画XY平面上的椭圆截面
        eigenvalues_3d, eigenvectors_3d = np.linalg.eigh(Sigma_3d[:2, :2])
        angle_3d = np.degrees(np.arctan2(eigenvectors_3d[1, 1], eigenvectors_3d[0, 1]))
        for n_std in [1, 2]:
            e = Ellipse(xy=(0,0),
                        width=2*n_std*np.sqrt(eigenvalues_3d[1]),
                        height=2*n_std*np.sqrt(eigenvalues_3d[0]),
                        angle=angle_3d, edgecolor='steelblue',
                        facecolor='steelblue', alpha=0.15*n_std, linewidth=2)
            ax3d.add_patch(e)
        # 画主轴
        for lam, vec in zip(eigenvalues_3d, eigenvectors_3d.T):
            ax3d.annotate('', xy=(np.sqrt(lam)*vec[0], np.sqrt(lam)*vec[1]),
                          xytext=(0,0),
                          arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax3d.set_xlim(-3, 3)
        ax3d.set_ylim(-3, 3)
        ax3d.set_aspect('equal')
        ax3d.grid(True, alpha=0.3)
        ax3d.set_title(f'{title}\n3D Σ（XY截面）', fontsize=8)

        # 下行：2D投影结果
        ax2d = axes[1, col]
        mu_2d = np.array([fx*mu_cam[0]/mu_cam[2], fy*mu_cam[1]/mu_cam[2]])
        xf = np.linspace(mu_2d[0]-80, mu_2d[0]+80, 150)
        yf = np.linspace(mu_2d[1]-80, mu_2d[1]+80, 150)
        Xf, Yf = np.meshgrid(xf, yf)
        pos = np.stack([Xf, Yf], axis=-1)
        diff = pos - mu_2d
        Sigma_inv = np.linalg.inv(Sigma_2d)
        Z = np.exp(-0.5 * np.einsum('...i,ij,...j', diff, Sigma_inv, diff))
        ax2d.contourf(Xf, Yf, Z, levels=10, cmap='Reds', alpha=0.7)
        ax2d.contour(Xf, Yf, Z, levels=5, colors='darkred', alpha=0.5)
        ax2d.plot(*mu_2d, 'b+', markersize=12, markeredgewidth=2)
        ax2d.set_title(f'2D投影 Σ\'', fontsize=8)
        ax2d.set_aspect('equal')
        ax2d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_05b_完整链路.png', dpi=120, bbox_inches='tight')
    print("\n已保存: output_05b_完整链路.png")
    plt.show()


# ─────────────────────────────────────────────
# 第三部分：数值稳定性对比
# ─────────────────────────────────────────────

def demo_numerical_stability():
    """
    对比：直接优化Σ vs 优化(q,s)的数值稳定性

    模拟梯度更新过程，展示直接优化Σ可能破坏正定性
    """
    print("\n" + "="*60)
    print("数值稳定性对比：直接优化Σ vs 优化(q,s)")
    print("="*60)

    np.random.seed(42)
    n_steps = 50
    lr = 0.1

    # 方法1：直接优化Σ的元素（危险！）
    Sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    eigenvalues_direct = []
    is_pd_direct = []

    for step in range(n_steps):
        # 模拟随机梯度（实际中梯度来自损失函数）
        grad = np.random.randn(2, 2) * 0.5
        grad = (grad + grad.T) / 2  # 保持对称
        Sigma = Sigma - lr * grad
        Sigma = (Sigma + Sigma.T) / 2  # 强制对称
        eigs = np.linalg.eigvalsh(Sigma)
        eigenvalues_direct.append(eigs.copy())
        is_pd_direct.append(np.all(eigs > 0))

    # 方法2：优化(q,s)，间接得到Σ（安全！）
    q = np.array([1.0, 0.0, 0.0, 0.0])  # 初始：无旋转
    s = np.array([np.sqrt(2.0), 1.0])   # 初始缩放
    eigenvalues_indirect = []
    is_pd_indirect = []

    for step in range(n_steps):
        # 模拟随机梯度更新q和s
        grad_q = np.random.randn(4) * 0.1
        grad_s = np.random.randn(2) * 0.1
        q = q - lr * grad_q
        q = q / np.linalg.norm(q)  # 归一化四元数
        s = np.abs(s - lr * grad_s)  # 保证s>0
        s = np.maximum(s, 0.01)

        R = quaternion_to_rotation_matrix(q)
        S = np.diag(s)
        Sigma_indirect = R[:2, :2] @ S @ S.T @ R[:2, :2].T
        eigs = np.linalg.eigvalsh(Sigma_indirect)
        eigenvalues_indirect.append(eigs.copy())
        is_pd_indirect.append(np.all(eigs > 0))

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("数值稳定性对比：直接优化Σ vs 优化(q,s)",
                 fontsize=12, fontweight='bold')

    steps = range(n_steps)
    eigs_direct = np.array(eigenvalues_direct)
    eigs_indirect = np.array(eigenvalues_indirect)

    # 特征值变化
    ax = axes[0]
    ax.plot(steps, eigs_direct[:, 0], 'r-', linewidth=2, label='直接优化Σ: λ₁')
    ax.plot(steps, eigs_direct[:, 1], 'r--', linewidth=2, label='直接优化Σ: λ₂')
    ax.axhline(0, color='k', linewidth=1, linestyle=':')
    ax.fill_between(steps, eigs_direct[:, 0], 0,
                    where=eigs_direct[:, 0] < 0, alpha=0.3, color='red',
                    label='负特征值（非正定！）')
    ax.set_title('直接优化Σ的特征值变化')
    ax.set_xlabel('训练步数')
    ax.set_ylabel('特征值')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(steps, eigs_indirect[:, 0], 'b-', linewidth=2, label='优化(q,s): λ₁')
    ax2.plot(steps, eigs_indirect[:, 1], 'b--', linewidth=2, label='优化(q,s): λ₂')
    ax2.axhline(0, color='k', linewidth=1, linestyle=':')
    ax2.set_title('优化(q,s)的特征值变化\n（始终为正！）')
    ax2.set_xlabel('训练步数')
    ax2.set_ylabel('特征值')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 正定性统计
    ax3 = axes[2]
    pd_rate_direct = np.cumsum(is_pd_direct) / (np.arange(n_steps) + 1)
    pd_rate_indirect = np.cumsum(is_pd_indirect) / (np.arange(n_steps) + 1)
    ax3.plot(steps, pd_rate_direct * 100, 'r-', linewidth=2, label='直接优化Σ')
    ax3.plot(steps, pd_rate_indirect * 100, 'b-', linewidth=2, label='优化(q,s)')
    ax3.set_title('正定性保持率（%）\n（越高越好）')
    ax3.set_xlabel('训练步数')
    ax3.set_ylabel('正定率 (%)')
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    n_pd_direct = sum(is_pd_direct)
    n_pd_indirect = sum(is_pd_indirect)
    print(f"\n直接优化Σ：{n_pd_direct}/{n_steps} 步保持正定 ({n_pd_direct/n_steps*100:.0f}%)")
    print(f"优化(q,s)：{n_pd_indirect}/{n_steps} 步保持正定 ({n_pd_indirect/n_steps*100:.0f}%)")
    print("→ 这就是3DGS选择(q,s)参数化的核心原因！")

    plt.tight_layout()
    plt.savefig('output_05c_数值稳定性.png', dpi=120, bbox_inches='tight')
    print("\n已保存: output_05c_数值稳定性.png")
    plt.show()


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Demo 05: 四元数与旋转 —— 3DGS参数化完整链路")
    print("=" * 60)

    demo_quaternion_geometry()
    demo_full_pipeline()
    demo_numerical_stability()

    print("\n" + "="*60)
    print("关键结论：")
    print("  1. 四元数用4个参数表示3D旋转，避免万向节锁")
    print("  2. 完整链路：q + s → R → Σ = RSSᵀRᵀ → Σ' = JΣJᵀ")
    print("  3. 优化(q,s)天然保证Σ正定，直接优化Σ会破坏正定性")
    print("  4. 官方代码：build_rotation() + get_covariance()")

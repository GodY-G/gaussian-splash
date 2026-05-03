import numpy as np

data = np.array([[1, 2], [2, 3], [3, 2], [2, 1], [1, 1]])
N = data.shape[0]

# 1. 计算均值 μ
mu = np.mean(data, axis=0)
print("均值 μ：", mu)

# 2. 手推协方差矩阵 Σ
centered_data = data - mu
Sigma = (centered_data.T @ centered_data) / N
print("协方差矩阵 Σ：\n", Sigma)

# 3. 计算 Σ 的特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(Sigma)
print("特征值：", eigenvalues)
print("特征向量：\n", eigenvectors)

# 4. 几何意义说明
print("\n--- 几何意义 ---")
print(f"主特征向量方向：{eigenvectors[:, np.argmax(eigenvalues)]}（数据分布的主方向）")
print(f"次特征向量方向：{eigenvectors[:, np.argmin(eigenvalues)]}（数据分布的次方向）")
print(f"主方向方差：{np.max(eigenvalues)}，次方向方差：{np.min(eigenvalues)}")
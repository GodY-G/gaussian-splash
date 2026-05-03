import numpy as np

A = np.array([[4, 2], [2, 4]])

# 1. 求特征值与特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print("特征值：", eigenvalues)
print("特征向量矩阵 Q：\n", eigenvectors)

# 2. 单位特征向量（numpy 已自动单位化）
v1 = eigenvectors[:, 0]
v2 = eigenvectors[:, 1]
print("对应λ1的单位特征向量：", v1)
print("对应λ2的单位特征向量：", v2)

# 3. 验证 A = QΛQᵀ
Lambda = np.diag(eigenvalues)
recon_A = eigenvectors @ Lambda @ eigenvectors.T
print("重构矩阵 A：\n", recon_A)
print("与原矩阵是否相等：", np.allclose(recon_A, A))

# 4. 验证 QᵀQ = I
print("Qᵀ·Q：\n", eigenvectors.T @ eigenvectors)
print("是否为单位矩阵：", np.allclose(eigenvectors.T @ eigenvectors, np.eye(2)))
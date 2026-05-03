import numpy as np

A = np.array([[2, 1], [0, 3]])

# 1. 手推验证：e1=(1,0) 和 e2=(0,1) 变换后
e1 = np.array([1, 0])
e2 = np.array([0, 1])
transformed_e1 = A @ e1
transformed_e2 = A @ e2
print("e1 变换后：", transformed_e1)
print("e2 变换后：", transformed_e2)

# 2. 行列式 det(A)
det_A = np.linalg.det(A)
print("det(A) =", det_A)
print("面积缩放比为 |det(A)| =", abs(det_A))

# 3. 逆矩阵 A⁻¹
A_inv = np.linalg.inv(A)
print("A⁻¹ =", A_inv)
# 验证 A·A⁻¹ = I
print("A·A⁻¹ =", A @ A_inv)

# 4. 验证 (A²)ᵀ = (Aᵀ)²
A_sq = A @ A
A_sq_T = A_sq.T
A_T = A.T
A_T_sq = A_T @ A_T
print("(A²)ᵀ =", A_sq_T)
print("(Aᵀ)² =", A_T_sq)
print("两者是否相等：", np.allclose(A_sq_T, A_T_sq))
import numpy as np

# 在3dgs中，每个高斯基元的位置就是一个3d向量
mu = np.array([1.5,0.3,4.2])

# 向量的模
v = np.array([3,4])
norm = np.linalg.norm(v)
print(norm)

v_hat = v / norm
print(v_hat)

# 向量的点积
a = np.array([3,0])
b1 = np.array([2,1])
b2 = np.array([0,2])
b3 = np.array([-2,1])

print(np.dot(a,b1))
print(np.dot(a,b2))
print(np.dot(a,b3))

# 叉积
a = np.array([3,0,0])
b = np.array([0,3,0])
print(np.cross(a,b))

c = np.cross(a,b)
print(np.dot(a,c))
print(np.dot(b,c))

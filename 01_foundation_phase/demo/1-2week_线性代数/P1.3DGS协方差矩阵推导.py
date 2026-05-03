import numpy as np
theta = np.radians(45)
R = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta),  np.cos(theta), 0],
              [0,              0,             1]])
S = np.diag([2.0, 0.5, 1.0])
Sigma = R @ S @ S.T @ R.T
print("特征值:", np.linalg.eigvalsh(Sigma))
import numpy as np
# 旋转与缩放
R = np.array([[0,-1],
              [1,0]])
result_R = R @ np.array([1,0])
print(result_R)

S = np.array([2,0],
             [0,2])
result_S = S @ np.array([1,0])
print(result_S)

# 平移：升维，齐次坐标法



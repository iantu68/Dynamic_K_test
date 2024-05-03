import numpy as np

data = [[1, 2], [2, 4]]
matrices = [np.array(sublist).reshape(1, 2) for sublist in data]

# 打印每个矩阵
for matrix in matrices:
    print(matrix)

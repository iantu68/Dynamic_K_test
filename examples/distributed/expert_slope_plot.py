import numpy as np
import matplotlib.pyplot as plt

# 读取数据文件
for i in range(8):
    with open(f'expert_grads_{i}.txt', 'r') as file:
        lines = file.readlines()

    # 解析数据并转换为数字值
    values = [float(line.split("(")[1].split(")")[0]) for line in lines]
    # print(values)
    # 計算相鄰值之間的差異
    differences = np.diff(values)

    # 計算斜率（兩兩值之間的斜率）
    slopes = differences / np.diff(range(len(values)))
    # 绘制连续点的图
    plt.figure()
    plt.plot(slopes)
    # plt.scatter(range(len(values)), values, color='b', marker='.', s=10)
    plt.title(f'Layer_0_exp_{i} Gradients Slope Values')
    plt.xlabel('Training Step')
    plt.ylabel('Gradients Value')
    plt.savefig(f'layer_0_exp_{i}_slope.png')
    plt.show()
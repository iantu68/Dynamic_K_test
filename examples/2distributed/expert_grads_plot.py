import numpy as np
import matplotlib.pyplot as plt

# 读取数据文件
for i in range(8):
    with open(f"expert_grads_{i}_L1_mean_first_nabs", 'r') as file:
        lines1 = file.readlines()
    with open(f"expert_grads_{i}_L1_mean_first_abs", 'r') as file:
        lines2 = file.readlines()
    # with open(f"expert_grads_{i}_L2_sub_first_nabs", 'r') as file:
    #     lines3 = file.readlines()
    # with open(f"expert_grads_{i}_L2_sub_first_abs", 'r') as file:
    #     lines4 = file.readlines()

    # 解析数据并转换为数字值
    values1 = [float(line.split("(")[1].split(")")[0]) for line in lines1]
    values2 = [float(line.split("(")[1].split(")")[0]) for line in lines2]
    # values3 = [float(line.split("(")[1].split(")")[0]) for line in lines3]
    # values4 = [float(line.split("(")[1].split(")")[0]) for line in lines4]
    # print(values)
    # 計算相鄰值之間的差異
    # differences = np.diff(values)

    # # 計算斜率（兩兩值之間的斜率）
    # slopes = differences / np.diff(range(len(values)))
    # 绘制连续点的图
    plt.figure()
    plt.plot(values1, label='L1_mfna')
    # plt.plot(values2, label='L1_mfa')
    plt.plot(values3, label='L1_sfna')
    # plt.plot(values4, label='L1_sfa')
    # plt.scatter(range(len(values)), values, color='b', marker='.', s=10)
    plt.title(f'L1_sub_first_abs')
    plt.xlabel('Training Step')
    plt.ylabel('Gradients Value')
    plt.savefig(f'L1_sub_first_abs_{i}_slope.png')
    plt.show()
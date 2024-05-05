import numpy as np
import matplotlib.pyplot as plt

# 读取数据文件
for i in range(8):
    # with open(f"expert_grads_{i}_L1_mean_first_nabs", 'r') as file:
    #     lines1 = file.readlines()
    # with open(f"expert_grads_{i}_L1_mean_first_abs", 'r') as file:
    #     lines2 = file.readlines()
    # with open(f"expert_grads_{i}_L1_sub_first_nabs", 'r') as file:
    #     lines3 = file.readlines()
    # with open(f"expert_grads_{i}_L1_sub_first_abs", 'r') as file:
    #     lines4 = file.readlines()
    # with open(f"expert_grads_{i}_L1_nabs.txt", 'r') as file:
    #     lines5 = file.readlines()
    # with open(f"expert_grads_{i}_L1_abs.txt", 'r') as file:
    #     lines6 = file.readlines()
    # with open(f"expert_grads_{i}_L2_nabs.txt", 'r') as file:
    #     lines7 = file.readlines()
    # with open(f"expert_grads_{i}_L2_abs.txt", 'r') as file:
    #     lines8 = file.readlines()

    # 解析数据并转换为数字值
    # values1 = [float(line.split("(")[1].split(")")[0]) for line in lines1]
    # values2 = [float(line.split("(")[1].split(")")[0]) for line in lines2]
    # values3 = [float(line.split("(")[1].split(")")[0]) for line in lines3]
    # values4 = [float(line.split("(")[1].split(")")[0]) for line in lines4]

    # values5 = [float(line.split("(")[1].split(")")[0]) for line in lines5]
    # # values6 = [float(line.split("(")[1].split(")")[0]) for line in lines6]
    # values7 = [float(line.split("(")[1].split(")")[0]) for line in lines7]
    # values8 = [float(line.split("(")[1].split(")")[0]) for line in lines8]
    # print(values)
    # 計算相鄰值之間的差異
    # differences = np.diff(values)


    values5 = np.load(f"expert_grads_{i}_L1_nabs.npy", allow_pickle=True)
    values7 = np.load(f"expert_grads_{i}_L2_nabs.npy", allow_pickle=True)
    print(values5)
    # # 計算斜率（兩兩值之間的斜率）
    # slopes = differences / np.diff(range(len(values)))
    # 绘制连续点的图
    plt.figure()
    # plt.plot(values1, label='L1_mfna')
    # plt.plot(values2, label='L1_mfa')
    # plt.plot(values3, label='L1_sfna')
    # plt.plot(values4, label='L1_sfa')
    plt.plot(values5, label='L1_na', color='red', linestyle='--')
    # plt.plot(values6, label='L1_a', color='red', linestyle='-')
    plt.plot(values7, label='L2_na', color='blue', linestyle='--')
    # plt.plot(values8, label='L2_a', color='blue', linestyle='-')
    # plt.scatter(range(len(values)), values, color='b', marker='.', s=10)
    plt.legend(loc='upper right')
    plt.title(f'L1_L2_grads_values')
    plt.xlabel('Training Step')
    plt.ylabel('Gradients Value')
    plt.ylim(0, 0.1)
    plt.savefig(f'L1_L2_grads_values_{i}.png')
    plt.show()
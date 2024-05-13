import numpy as np
import matplotlib.pyplot as plt

# 读取数据文件
for i in range(8):
    values1 = np.load(f"expert_grads_L0_FFN0_{i}_nabs.npy", allow_pickle=True)
    values2 = np.load(f"expert_grads_L0_FFN1_{i}_nabs.npy", allow_pickle=True)
    print(values1)
    # # 計算斜率（兩兩值之間的斜率）
    # slopes = differences / np.diff(range(len(values)))
    # 绘制连续点的图
    plt.figure()
    # plt.plot(values1, label='L1_mfna')
    # plt.plot(values2, label='L1_mfa')
    # plt.plot(values3, label='L1_sfna')
    # plt.plot(values4, label='L1_sfa')
    plt.plot(values1, label='L0_FFN0_na', color='red', linestyle='--')
    # plt.plot(values6, label='L1_a', color='red', linestyle='-')
    plt.plot(values2, label='L0_FFN1_na', color='blue', linestyle='--')
    # plt.plot(values8, label='L2_a', color='blue', linestyle='-')
    # plt.scatter(range(len(values)), values, color='b', marker='.', s=10)
    plt.legend(loc='upper right')
    plt.title(f'L0_grads_values')
    plt.xlabel('Training Step')
    plt.ylabel('Gradients Value')
    plt.ylim(0, 0.1)
    plt.savefig(f'L0_grads_values{i}.png')
    plt.show()
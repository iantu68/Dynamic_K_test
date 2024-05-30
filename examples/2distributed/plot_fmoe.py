import numpy as np
import matplotlib.pyplot as plt

# 載入檔案
loss2 = np.load('losses.npy')
acc = np.load('acc.npy')
frequency_0 = np.load('expert_counts_layer_0.npy')
frequency_1 = np.load('expert_counts_layer_1.npy')

# 計算 sigmoid 值
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 計算每一步的平均值
def calculate_avg_steps(data):
    each_step_avg = []
    total_steps = 0
    for i, steps in enumerate(data):
        total_steps += steps
        avg_steps = total_steps / (i + 1)
        each_step_avg.append(avg_steps)
    return each_step_avg

# 計算avg_sigmoid_all_average點與點之間的斜率
def calculate_slopes(y_values):
    slopes = []
    for i in range(1, len(y_values)):
        x1, y1 = i - 1, y_values[i - 1]
        x2, y2 = i, y_values[i]
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)
    return slopes

# 设置直方图的边界和中心
bin_edges = np.arange(len(frequency_1) + 1)
bin_centers = bin_edges[:-1]

# 每个条形的宽度
bar_width = 0.5


# --------------------------------------------------------------------------------------------
# Plot Experts Frequency
plt.figure()
plt.bar(bin_centers + bar_width, frequency_0, color='blue', width=bar_width, label='Layer 0')
plt.xlabel('Expert Index')
plt.ylabel('Counts')
plt.title('Expert Counts by Layer')
plt.xticks(bin_centers + bar_width / 2, range(len(frequency_0)))
plt.legend(loc='upper right')
plt.savefig('Expert_count_Layer0.png')
plt.show()


plt.figure()
plt.bar(bin_centers + bar_width, frequency_1, color='blue', width=bar_width, label='Layer 0')
plt.xlabel('Expert Index')
plt.ylabel('Counts')
plt.title('Expert Counts by Layer')
plt.xticks(bin_centers + bar_width / 2, range(len(frequency_1)))
plt.legend(loc='upper right')
plt.savefig('Expert_count_Layer1.png')
plt.show()

# --------------------------------------------------------------------------------------------
# Plot Loss
loss2 = calculate_avg_steps(loss2)

plt.figure()
plt.plot(loss2, label='Train_loss', color='blue')
plt.legend(loc='upper right')
plt.title(f'Loss_Values_Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.savefig(f'Loss_Values_Curve.png')
plt.show()

# --------------------------------------------------------------------------------------------
# Plot Acc
acc = calculate_avg_steps(acc)

plt.figure()
plt.plot(acc, label='Accuracy', color='red', linestyle='-')
max_acc = max(acc)
max_acc_epoch = acc.index(max_acc)
plt.axvline(x=max_acc_epoch, color='red', linewidth=0.8)
plt.annotate(f'Max Accuracy: {max_acc:.4f} at Epoch {max_acc_epoch}', 
             xy=(max_acc_epoch, max_acc), 
             xytext=(max_acc_epoch, max_acc + 0.1))
plt.legend(loc='upper right')
plt.title(f'Accuracy_Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.savefig(f'Accuracy_Curve.png')
plt.show()



# 读取数据文件
for i in range(8):
    values1 = np.load(f"expert_grads_L0_FFN0_{i}_nabs.npy", allow_pickle=True)
    values2 = np.load(f"expert_grads_L0_FFN1_{i}_nabs.npy", allow_pickle=True)
    values3 = np.load(f"expert_grads_L1_FFN0_{i}_nabs.npy", allow_pickle=True)
    values4 = np.load(f"expert_grads_L1_FFN1_{i}_nabs.npy", allow_pickle=True)
    # values5 = np.load(f"expert_grads_L2_FFN0_{i}_nabs.npy", allow_pickle=True)
    # values6 = np.load(f"expert_grads_L2_FFN1_{i}_nabs.npy", allow_pickle=True)
    # values7 = np.load(f"expert_grads_L3_FFN0_{i}_nabs.npy", allow_pickle=True)
    # values8 = np.load(f"expert_grads_L3_FFN1_{i}_nabs.npy", allow_pickle=True)
    
    plt.figure()
    plt.plot(values1, label='L0_FFN0_na', color='red', linestyle='--')
    plt.plot(values2, label='L0_FFN1_na', color='blue', linestyle='--')
    plt.legend(loc='upper right')
    plt.title(f'L1_grads_values')
    plt.xlabel('Training Step')
    plt.ylabel('Gradients Value')
    plt.ylim()
    plt.savefig(f'L0_grads_values{i}.png')
    plt.show()

    plt.figure()
    plt.plot(values3, label='L1_FFN0_na', color='red', linestyle='--')
    plt.plot(values4, label='L1_FFN1_na', color='blue', linestyle='--')
    plt.legend(loc='upper right')
    plt.title(f'L1_grads_values')
    plt.xlabel('Training Step')
    plt.ylabel('Gradients Value')
    plt.ylim()
    plt.savefig(f'L1_grads_values{i}.png')
    plt.show()

    # plt.figure()
    # plt.plot(values5, label='L2_FFN0_na', color='red', linestyle='--')
    # plt.plot(values6, label='L2_FFN1_na', color='blue', linestyle='--')
    # plt.legend(loc='upper right')
    # plt.title(f'L1_grads_values')
    # plt.xlabel('Training Step')
    # plt.ylabel('Gradients Value')
    # plt.ylim()
    # plt.savefig(f'L2_grads_values{i}.png')
    # plt.show()

    # plt.figure()
    # plt.plot(values7, label='L3_FFN0_na', color='red', linestyle='--')
    # plt.plot(values8, label='L3_FFN1_na', color='blue', linestyle='--')
    # plt.legend(loc='upper right')
    # plt.title(f'L1_grads_values')
    # plt.xlabel('Training Step')
    # plt.ylabel('Gradients Value')
    # plt.ylim()
    # plt.savefig(f'L3_grads_values{i}.png')
    # plt.show()


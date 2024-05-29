import numpy as np
import matplotlib.pyplot as plt

# 載入檔案
data0 = np.load('Layer0_grads.npy')[:1000]
data1 = np.load('Layer1_grads.npy')[:1000]
data2 = np.load('Layer2_grads.npy')[:1000]
data3 = np.load('Layer3_grads.npy')[:1000]
loss1 = np.load('Valid_Loss.npy')[:1000]
loss2 = np.load('Train_Loss.npy')[:1000]
acc = np.load('Accuracy.npy')[:1000]

# 確保所有數據的長度一致
min_length = min(len(data0), len(data1), len(data2), len(data3))
data0 = data0[:min_length]
data1 = data1[:min_length]
data2 = data2[:min_length]
data3 = data3[:min_length]

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

sigmoid_data0 = sigmoid(data0)
sigmoid_data1 = sigmoid(data1)
sigmoid_data2 = sigmoid(data2)
sigmoid_data3 = sigmoid(data3)

avg_steps_data0 = calculate_avg_steps(data0)
avg_steps_data1 = calculate_avg_steps(data1)
avg_steps_data2 = calculate_avg_steps(data2)
avg_steps_data3 = calculate_avg_steps(data3)
loss1 = calculate_avg_steps(loss1)
loss2 = calculate_avg_steps(loss2)
acc = calculate_avg_steps(acc)

avg_steps_data0_slopes = calculate_slopes(avg_steps_data0)
avg_steps_data1_slopes = calculate_slopes(avg_steps_data1)
avg_steps_data2_slopes = calculate_slopes(avg_steps_data2)
avg_steps_data3_slopes = calculate_slopes(avg_steps_data3)

# 計算同個時間點，不同層之間的平均
all_slopes_average = []
for i in range(len(avg_steps_data0_slopes)):
    value1 = avg_steps_data0_slopes[i] + avg_steps_data1_slopes[i] + avg_steps_data2_slopes[i] + avg_steps_data3_slopes[i]
    avg_value1 = value1 / 4
    all_slopes_average.append(avg_value1)

# --------------------------------------------------------------------------------------------
# Plot Grads_Values
plt.figure()
plt.plot(avg_steps_data0, label='Layer_0', color='red', linestyle='-')
plt.plot(avg_steps_data1, label='Layer_1', color='blue', linestyle='-')
plt.plot(avg_steps_data2, label='Layer_2', color='green', linestyle='-')
plt.plot(avg_steps_data3, label='Layer_3', color='purple', linestyle='-')
plt.legend(loc='upper right')
plt.title(f'MLP_Grads_Values')
plt.xlabel('Epoch')
plt.ylabel('Gradients Value')
plt.savefig(f'MLP_Grads_Values.png')
plt.show()

# --------------------------------------------------------------------------------------------
# Plot all_slope
plt.figure()
plt.plot(avg_steps_data0_slopes , label='Layer_0 Avg', color='red')
plt.plot(avg_steps_data1_slopes , label='Layer_1 Avg', color='blue')
plt.plot(avg_steps_data2_slopes , label='Layer_2 Avg', color='green')
plt.plot(avg_steps_data3_slopes , label='Layer_3 Avg', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Slope')
plt.title('Slopes')
plt.legend()
plt.grid(True)
plt.savefig(f'All_Grads_slop.png')
plt.show()

# --------------------------------------------------------------------------------------------
# Plot Avg_slope
plt.figure()
plt.plot(all_slopes_average , label='Data Slopes', color='black')
plt.axhline(y=0.001, color='r', linestyle='-', linewidth=0.8)
plt.text(0, 0.001, '0.001', color='r', va='bottom', ha='right')
plt.xlabel('Epoch')
plt.ylabel('Slope')
# plt.ylim(0, 0.02)
plt.title('Slopes')
plt.legend()
plt.grid(True)
plt.savefig(f'Avg_all_Grads_slop.png')
plt.show()

# --------------------------------------------------------------------------------------------
# Plot Loss
plt.figure()
plt.plot(loss1, label='Valid_loss', color='red')
plt.plot(loss2, label='Train_loss', color='blue')
min_valid_loss = min(loss1)
min_valid_epoch = loss1.index(min_valid_loss)
plt.axvline(x=min_valid_epoch, color='red', linestyle='-', linewidth=0.8)
plt.annotate(f'Min Valid Loss: {min_valid_loss:.4f} at Epoch {min_valid_epoch}', 
             xy=(min_valid_epoch, min_valid_loss), 
             xytext=(min_valid_epoch, min_valid_loss + 0.1))
plt.legend(loc='upper right')
plt.title(f'Loss_Values_Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.savefig(f'Loss_Values_Curve.png')
plt.show()

# --------------------------------------------------------------------------------------------
# Plot Acc
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

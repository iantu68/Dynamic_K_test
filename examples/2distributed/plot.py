import numpy as np
import matplotlib.pyplot as plt

# 載入.npy檔案
data0 = np.load('Layer0_grads.npy')
data1 = np.load('Layer1_grads.npy')
data2 = np.load('Layer2_grads.npy')
data3 = np.load('Layer3_grads.npy')
loss1 = np.load('Valid_Loss.npy')
loss2 = np.load('Train_Loss.npy')
acc = np.load('Accuracy.npy')


# 计算每层的平均梯度值
avg_grad0 = np.mean(data0)
avg_grad1 = np.mean(data1)
avg_grad2 = np.mean(data2)
avg_grad3 = np.mean(data3)

# --------------------------------------------------------------------------------------------
# Plot Grads
plt.figure()
plt.plot(data0, label='Layer_0', color='red', linestyle='-')
plt.plot(data1, label='Layer_1', color='blue', linestyle='-')
plt.plot(data2, label='Layer_2', color='green', linestyle='-')
plt.plot(data3, label='Layer_3', color='purple', linestyle='-')

# 绘制平均梯度的水平线
plt.axhline(y=avg_grad0, color='red', linestyle='--', linewidth=0.5)
plt.axhline(y=avg_grad1, color='blue', linestyle='--', linewidth=0.5)
plt.axhline(y=avg_grad2, color='green', linestyle='--', linewidth=0.5)
plt.axhline(y=avg_grad3, color='purple', linestyle='--', linewidth=0.5)

# 标记平均梯度值
plt.text(0, avg_grad0, f'{avg_grad0:.4f}', color='red', va='bottom', ha='right')
plt.text(0, avg_grad1, f'{avg_grad1:.4f}', color='blue', va='bottom', ha='right')
plt.text(0, avg_grad2, f'{avg_grad2:.4f}', color='green', va='bottom', ha='right')
plt.text(0, avg_grad3, f'{avg_grad3:.4f}', color='purple', va='bottom', ha='right')

plt.legend(loc='upper right')
plt.title(f'MLP_Grads_Values')
plt.xlabel('Epoch')
plt.ylabel('Gradients Value')
plt.savefig(f'MLP_Grads_Values.png')
plt.show()

# --------------------------------------------------------------------------------------------
# Plot Loss
plt.figure()
plt.plot(loss1, label='Valid_loss', color='red', linestyle='--')
plt.plot(loss2, label='Train_loss', color='blue', linestyle='--')

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

plt.legend(loc='upper right')
plt.title(f'Accuracy_Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.savefig(f'Accuracy_Curve.png')
plt.show()
import numpy as np
import matplotlib.pyplot as plt


# 載入.npy檔案
data0 = np.load('Layer0_grads.npy')
data1 = np.load('Layer1_grads.npy')
data2 = np.load('Layer2_grads.npy')
loss1 = np.load('Valid_Loss.npy')
loss2 = np.load('Train_Loss.npy')

plt.figure()
plt.plot(data0, label='Layer_0', color='red', linestyle='--')
# plt.plot(data1, label='Layer_1', color='blue', linestyle='--')
# plt.plot(data2, label='Layer_2', color='green', linestyle='--')

plt.legend(loc='upper right')
plt.title(f'MLP_Grads_Values')
plt.xlabel('Epoch')
plt.ylabel('Gradients Value')
# plt.ylim(0, 0.1)
plt.savefig(f'MLP_Grads_Values_0.png')
plt.show()

#Plot Loss
plt.figure()
plt.plot(loss1, label='Valid_loss', color='red', linestyle='--')
plt.plot(loss2, label='Train_loss', color='blue', linestyle='--')

plt.legend(loc='upper right')
plt.title(f'Loss_Values_Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
# plt.ylim(0, 0.1)
plt.savefig(f'Loss_Values_Curve.png')
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# 讀取損失值文件
with open('loss_value.txt', 'r') as file:
    lines = file.readlines()

# 將每行的數字轉換為浮點數
loss_values = [float(line.strip()) for line in lines]

# 繪製損失值曲線
plt.plot(loss_values, color='b')
plt.title('Loss Values Curve')
plt.xlabel('Training Step')
plt.ylabel('Loss value')
plt.savefig('Loss_value.png')
plt.show()
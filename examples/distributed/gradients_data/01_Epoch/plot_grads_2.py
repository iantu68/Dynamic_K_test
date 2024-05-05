import numpy as np
import matplotlib.pyplot as plt

# 讀取資料
with open('expert_grads_0_L1_nabs.txt', 'r') as file:
    lines = file.readlines()

# 將資料轉換為 numpy 陣列
data = np.array([eval(line.strip()) for line in lines])
# 計算相鄰向量的差異
diff_data = np.diff(data, axis=0)
# 計算平均值
mean_diff = np.abs(mean(diff_data, axis=0))

# 繪製平均值
plt.bar(range(len(mean_diff)), mean_diff)
plt.xlabel('Index')
plt.ylabel('Mean Absolute Difference')
plt.title('Mean Absolute Difference of Vector Components')
plt.savefig('mean_absolute_difference_plot.png')  # 將繪製的圖保存為PNG格式
plt.show()
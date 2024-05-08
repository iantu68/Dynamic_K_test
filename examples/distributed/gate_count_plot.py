import numpy as np
import matplotlib.pyplot as plt

# 两个数组
experts_counts_layer_0 = [29452437, 25415972, 81275011, 25957833, 52637311, 14623946, 11919392, 62342138]
experts_counts_layer_1 = [46226284, 52625355, 15057588, 20959712, 40011511, 24212363, 83988101, 20543126]

# 设置直方图的边界和中心
bin_edges = np.arange(len(experts_counts_layer_0) + 1)
bin_centers = bin_edges[:-1]

# 每个条形的宽度
bar_width = 0.4

# 绘制直方图
plt.bar(bin_centers, experts_counts_layer_0, color='blue', alpha=0.5, width=bar_width, label='Layer 0')
# plt.bar(bin_centers + bar_width, experts_counts_layer_1, color='red', alpha=0.5, width=bar_width, label='Layer 1')

# 添加标签和标题
plt.xlabel('Expert Index')
plt.ylabel('Counts')
plt.title('Expert Counts by Layer')
plt.xticks(bin_centers + bar_width / 2, range(len(experts_counts_layer_0)))

# 添加图例
plt.legend(loc='upper right')

# 保存图像
plt.savefig('Expert_count.png')

# 显示图形
plt.show()
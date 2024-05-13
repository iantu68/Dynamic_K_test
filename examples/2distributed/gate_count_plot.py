import numpy as np
import matplotlib.pyplot as plt

# 两个数组
experts_counts_layer_0=[5200154, 4439297, 20364888, 3595065, 10742478, 5274477, 1538936, 9569513]
experts_counts_layer_1=[5886269, 14279243, 2723314, 2159586, 8434003, 8827661, 11437242, 6977490]

# 设置直方图的边界和中心
bin_edges = np.arange(len(experts_counts_layer_1) + 1)
bin_centers = bin_edges[:-1]

# 每个条形的宽度
bar_width = 0.5

# 绘制直方图
# plt.bar(bin_centers, experts_counts_layer_0, color='blue', width=bar_width, label='Layer 0')
plt.bar(bin_centers + bar_width, experts_counts_layer_0, color='blue', width=bar_width, label='Layer 1')

# 添加标签和标题
plt.xlabel('Expert Index')
plt.ylabel('Counts')
plt.title('Expert Counts by Layer')
plt.xticks(bin_centers + bar_width / 2, range(len(experts_counts_layer_1)))

# 添加图例
plt.legend(loc='upper right')

# 保存图像
plt.savefig('Expert_count_Layer0.png')

# 显示图形
plt.show()
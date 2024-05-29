import numpy as np
import matplotlib.pyplot as plt

# 两个数组
experts_counts_layer_0=[212238, 243058, 6580192, 1019942, 7848537, 12242600, 430302, 1785535]
experts_counts_layer_1=[3490002, 7554422, 2382630, 1840093, 101257, 6302927, 4344679, 4346394]


# experts_counts_layer_2=[1675594, 210513769, 2833833967, 553393, 117793, 3025230562, 431934, 393388]
# experts_counts_layer_3=[1387944306, 459598, 1772654747, 462474, 1301171650, 1609088562, 185810, 783253]



# 设置直方图的边界和中心
bin_edges = np.arange(len(experts_counts_layer_1) + 1)
bin_centers = bin_edges[:-1]

# 每个条形的宽度
bar_width = 0.5

plt.figure()
# 绘制直方图
# plt.bar(bin_centers, experts_counts_layer_0, color='blue', width=bar_width, label='Layer 0')
plt.bar(bin_centers + bar_width, experts_counts_layer_0, color='blue', width=bar_width, label='Layer 0')
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


plt.figure()
plt.bar(bin_centers + bar_width, experts_counts_layer_1, color='blue', width=bar_width, label='Layer 1')
# 添加标签和标题
plt.xlabel('Expert Index')
plt.ylabel('Counts')
plt.title('Expert Counts by Layer')
plt.xticks(bin_centers + bar_width / 2, range(len(experts_counts_layer_1)))
# 添加图例
plt.legend(loc='upper right')
# 保存图像
plt.savefig('Expert_count_Layer1.png')
# 显示图形
plt.show()

# plt.figure()
# plt.bar(bin_centers + bar_width, experts_counts_layer_2, color='blue', width=bar_width, label='Layer 1')
# # 添加标签和标题
# plt.xlabel('Expert Index')
# plt.ylabel('Counts')
# plt.title('Expert Counts by Layer')
# plt.xticks(bin_centers + bar_width / 2, range(len(experts_counts_layer_1)))
# # 添加图例
# plt.legend(loc='upper right')
# # 保存图像
# plt.savefig('Expert_count_Layer2.png')
# # 显示图形
# plt.show()

# plt.figure()
# plt.bar(bin_centers + bar_width, experts_counts_layer_3, color='blue', width=bar_width, label='Layer 1')
# # 添加标签和标题
# plt.xlabel('Expert Index')
# plt.ylabel('Counts')
# plt.title('Expert Counts by Layer')
# plt.xticks(bin_centers + bar_width / 2, range(len(experts_counts_layer_1)))
# # 添加图例
# plt.legend(loc='upper right')
# # 保存图像
# plt.savefig('Expert_count_Layer3.png')
# # 显示图形
# plt.show()
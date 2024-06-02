import numpy as np
import matplotlib.pyplot as plt

# 計算兩點之間線段角度的函數
def calculate_angle(x1, y1, x2, y2):
    # 計算向量的方向角
    angle_radians = np.arctan2(y2 - y1, x2 - x1)
    # 將弧度轉換為角度
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

# 示例點
points = [(1, 3), (2, 2), (3, 1), (4, 0), (5, -1), (6, -2), (7, -3), (8, -4), (9, -5), (10, -6)]

angles = []
for i in range(len(points) - 1):
    x1, y1 = points[i]
    x2, y2 = points[i+1]
    angle = calculate_angle(x1, y1, x2, y2)
    angles.append(angle)

# 繪製點和連接線
x_coords, y_coords = zip(*points)

plt.figure(figsize=(10, 6))
plt.plot(x_coords, y_coords, 'bo-', label='Points')
for i, angle in enumerate(angles):
    plt.text((x_coords[i] + x_coords[i+1]) / 2, (y_coords[i] + y_coords[i+1]) / 2, f'{angle:.2f}°', 
             fontsize=9, color='red', ha='center')

# 添加標題和標籤
plt.title('Points and Angles Between Consecutive Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

# 保存為PNG文件
plt.savefig('points_and_angles.png')  # 替換成你想保存的路徑

# 顯示圖表
plt.show()

print("Points:", points)
print("Angles between consecutive points:", angles)

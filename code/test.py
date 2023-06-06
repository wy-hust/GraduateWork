import matplotlib.pyplot as plt

# 准备数据
x = [1, 2, 3, 4, 5]
y = [10, 8, 6, 4, 2]

# 绘制折线图
plt.plot(x, y)

# 添加标题和坐标轴标签
plt.title("Line Chart")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

# 显示图表
plt.show()
import matplotlib.pyplot as plt  # 导入绘图模块
from mpl_toolkits.mplot3d import Axes3D  # 3d绘图模块
import numpy as np  # 导入数值计算拓展模块

# start generating points
x_lim = np.linspace(-10, 10, 220)
y_lim = np.linspace(-10, 10, 220)
z_lim = np.linspace(-10, 10, 220)
X_points = []  # 用来存放绘图点X坐标
Y_points = []  # 用来存放绘图点Y坐标
Z_points = []  # 用来存放绘图点Z坐标
for x in x_lim:
    for y in y_lim:
        for z in z_lim:
            if (x ** 2 + (9 / 4) * y ** 2 + z ** 2 - 1) ** 3 - (9 / 80) * y ** 2 * z ** 3 - x ** 2 * z ** 3 <= 0:
                X_points.append(x)
                Y_points.append(y)
                Z_points.append(z)

###start plot love
fig = plt.figure()  # 画布初始化
ax = fig.add_subplot(111, projection='3d')  # 采用3d绘图
ax.scatter(X_points, Y_points, Z_points, s=20, alpha=0.5, color="k")  # 3d散点图填充
plt.show()

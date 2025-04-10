import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Tạo dữ liệu
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
num_points = 50
x = np.linspace(-2, 2, num_points)
y = np.linspace(-2, 2, num_points)
X, Y = np.meshgrid(x, y)
Z = np.square(X) + np.square(Y)

# Vẽ bề mặt
surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0.5, edgecolors='k')

# Vẽ đường đồng mức trên mặt phẳng xy
cset = ax.contour(X, Y, Z, zdir='z', offset=-1.5, cmap=cm.coolwarm)

# Đặt giới hạn trục
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1.5, 5)

# Thiết lập nhãn và tiêu đề
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$y$', fontsize=14)
ax.set_zlabel('$f(x, y) = |x| + |y|$', fontsize=14)

plt.show()
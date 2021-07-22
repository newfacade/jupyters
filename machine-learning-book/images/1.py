import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

"""
x = np.linspace(0.0, 1.0, 101)
y = -x * np.log(x + 1e-10) - (1 - x) * np.log(1 - x + 1e-10)
plt.plot(x, y, label='Entropy')
# y2 = x * (1 - x) * 4 * np.log(2)
# plt.plot(x, y2, label='Gini after amplify')
plt.title("Entropy")
plt.legend()
plt.savefig("entropy.png", dpi=300, format='png')
plt.show()
"""

"""
points = np.array([[np.random.rand(), np.random.rand()] for _ in range(20)])
plt.scatter(points[:, 0], points[:, 1])
y1 = np.percentile(points[:, 1], 50)
plt.axhline(y1)
subset1 = points[points[:, 1] >= y1]
subset2 = points[points[:, 1] < y1]
x1 = np.percentile(subset1[:, 0], 50)
x2 = np.percentile(subset2[:, 0], 50)
plt.vlines(x1, y1, 1)
plt.vlines(x2, 0, y1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig("kdtree.png", dpi=300, format="png")
plt.show()
"""

"""
fig, ax = plt.subplots(figsize=(5, 5))

circle = Circle((0.14, 0.98), radius=0.5, facecolor='white', edgecolor='cornflowerblue', linewidth=2)
ax.add_patch(circle)
ax.set_xlim(-1, 1)
ax.set_ylim(-0.5, 1.6)
l1 = 0.5
x = np.linspace(0, l1, 100)
ax.plot(x, l1 - x, c='r')
ax.plot(-x, l1 - x, c='r')
ax.plot(x, x - l1, c='r')
ax.plot(-x, x - l1, c='r')
ax.axvline(0, c='black')
ax.axhline(0, c='black')
ax.text(-0.6, 0.2, "L1 term")
ax.text(-0.7, 1.0, "MSE loss")
ax.arrow(-0.3, 0.5, 0.3, 0, width=0.01,
         length_includes_head=True,  # 增加的长度包含箭头部分
         head_width=0.025,
         head_length=0.1)
ax.text(-0.57, 0.5, "optimal")
plt.savefig("lasso.png", dpi=300, format="png")
plt.show()
"""

a = np.random.multivariate_normal([0, 3], [[1, 0.5], [0.5, 1]], size=20)
x, y = [], []
for item in a:
    if item[1] >= item[0] + 1:
        x.append(item[0])
        y.append(item[1])
x.append(1)
y.append(2)
print(a.shape)
plt.scatter(x, y, c='blue', marker=".", label='positive')

b = np.random.multivariate_normal([3, 0], [[1, 0.5], [0.5, 1]], size=20)
x, y = [], []
for item in b:
    if item[1] <= item[0] - 1:
        x.append(item[0])
        y.append(item[1])
x.append(2)
y.append(1)
plt.scatter(x, y, c='red', marker="*", label='negative')

x = np.linspace(-2, 5, 100)
plt.plot(x, x, label='decision boundary')
plt.plot(x, x + 1, linestyle='--', c='green', label='margin boundary')
plt.plot(x, x - 1, linestyle='--', c='green')
plt.xlim(-2, 5)
plt.ylim(-2, 5)
plt.legend()
plt.savefig("margin.png", dpi=300, format='png')
plt.show()

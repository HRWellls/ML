import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 定义正例和负例数据
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
y = np.array([1, 1, 1, -1, -1])

# 使用SVM进行训练
clf = svm.SVC(kernel='linear', C=5)
clf.fit(X, y)

# 获取超平面参数
w = clf.coef_[0]
b = clf.intercept_[0]

print('w:', w)
print('b:', b)

# 计算支持向量
support_vectors = clf.support_vectors_

# 绘制数据点
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='o', label='Data Points')

# 绘制支持向量
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='black', label='Support Vectors')

# 绘制决策超平面
ax = plt.gca()
xx, yy = np.meshgrid(np.linspace(0, 4, 30), np.linspace(0, 4, 30))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界和间隔边界
plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.75, colors=['red', 'black', 'blue'], linestyles=['--', '-', '--'])

# 添加标签和标题
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Classification with Maximal Margin')
plt.legend()
plt.grid(True)
plt.show()


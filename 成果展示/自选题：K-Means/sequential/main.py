import numpy as np
from scipy.cluster.vq import kmeans, whiten
import matplotlib.pyplot as plt
import time
# 对数据进行标准化处理
obs = whiten(np.genfromtxt('data.csv', dtype=float, delimiter=','))
# 希望得到的聚类个数
K = 3
# K-means聚类尝试的次数
nstart = 100000
# 设置随机种子为0，每次运行代码时生成的随机数都是相同的
np.random.seed(0)
# centroids是聚类中心，distortion是失真度，即数据点到其最近的聚类中心距离的平方和
t0 = time.time()
centroids, distortion = kmeans(obs, K, nstart)
t1 = time.time()
# 打印结果
print('Best distortion for %d tries: %f' % (nstart, distortion))
print('Running time: {0}'.format(t1-t0))
# 用图像展示聚类结果
plt.scatter(obs[:, 0], obs[:, 1],c = 'g')
plt.scatter(centroids[:, 0], centroids[:, 1], c='r')
plt.show()

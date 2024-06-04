import numpy as np
from scipy.cluster.vq import kmeans, whiten
from operator import itemgetter
from math import ceil
from mpi4py import MPI
import matplotlib.pyplot as plt
import time

# 默认的MPI通信器，代表所有进程
comm = MPI.COMM_WORLD
# 表示当前进程的编号p
rank = comm.Get_rank()
# 表示总的进程数
size = comm.Get_size()
# 设置随机种子为当前进程的编号，以确保每个进程使用不同的随机数序列
np.random.seed(seed=rank)
# 读取数据并进行标准化处理
obs = whiten(np.genfromtxt('data.csv', dtype=float, delimiter=','))
# 希望得到3个类
K = 3
# K聚类尝试的次数
nstart = 100000
# n表示每个进程要进行尝试的次数，通过将nstart均分给各个进程来确定
n = int(ceil(float(nstart)/size))
t0 = time.time()
# 进行K-means聚类并计算失真度
centroids, distortion = kmeans(obs, K, n)
# 收集所有进程的聚类结果，将结果发送到根进程
results = comm.gather((centroids, distortion), root=0)

# # 在根进程中，按失真度对结果进行排序，选择失真度最小的结果，打印出最佳失真度
# if rank == 0:
#     results.sort(key=itemgetter(1))
#     result = results[0]
#     t1 = time.time()
#     print('Best distortion for %d tries: %f' % (nstart, result[1]))
#     print('Running time: {0}'.format(t1 - t0))
#
# # 用图像展示聚类结果
# plt.scatter(obs[:, 0], obs[:, 1],c = 'g')
# plt.scatter(centroids[:, 0], centroids[:, 1], c='r')
# plt.show()

if rank == 0:
    # 在根进程中，按失真度对结果进行排序，选择失真度最小的结果，打印出最佳失真度
    results.sort(key=itemgetter(1))
    result = results[0]
    best_centroids = result[0]
    best_distortion = result[1]
    t1 = time.time()
    print('Best distortion for %d tries: %f' % (nstart, best_distortion))
    print('Running time: {0}'.format(t1 - t0))
else:
    best_centroids = None

# 广播最佳聚类中心
best_centroids = comm.bcast(best_centroids, root=0)

# 用图像展示聚类结果
if rank == 0:
    plt.scatter(obs[:, 0], obs[:, 1], c='g')
    plt.scatter(best_centroids[:, 0], best_centroids[:, 1], c='r')
    plt.show()




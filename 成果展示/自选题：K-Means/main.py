import csv
import numpy as np

# 设置随机种子以确保每次生成的数据都是相同的
np.random.seed(0)

# 设置簇的数量
num_clusters = 3
# 每个簇的样本数量
samples_per_cluster = 30000
# 每个簇的维度（特征数量）
num_features = 2

# 定义三个簇的中心位置
centroids = np.array([[2, 2], [7, 7], [-2, -2]])

# 初始化一个空列表来存储所有数据
all_data = []

# 为每个簇生成数据
for centroid in centroids:
    # 使用正态分布为每个簇生成样本
    # 这里的stdv表示标准偏差，可以调整以使簇之间的重叠更少
    stdv = 1.0
    cluster_data = np.random.normal(centroid, stdv, (samples_per_cluster, num_features))
    all_data.append(cluster_data)

# 将所有簇的数据合并为一个numpy数组
all_data = np.vstack(all_data)

# 将数据写入CSV文件
with open('data1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # 写入数据
    for row in all_data:
        writer.writerow(row)

print("data1.csv文件已生成！")
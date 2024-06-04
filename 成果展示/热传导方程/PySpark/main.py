"""
基于两个 准备的代码
"D:\myCodes\HPCbook\05pyspark\sparkTest1NumpyMattoRDD9.py"
"D:\myCodes\HPCbook\05pyspark\sparkTest1NumpyMattoRDD10.py"
"""

from __future__ import print_function
import numpy as np
import timeit
# =============matplotlib 配置 =========================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set the colormap
# plt.rcParams['image.cmap'] = 'BrBG'
plt.rcParams['image.cmap'] = 'jet'  # you can try: colourMap = plt.cm.coolwarm
plt.figure(dpi=300)
# Set colour interpolation and colour map
colorinterpolation = 100
colourMap = plt.cm.jet  # you can try: colourMap = plt.cm.coolwarm
# Set Dimension
lenX = lenY = 400  # we set it rectangular
# Set meshgrid
X, Y = np.meshgrid(np.arange(0, lenX), np.arange(0, lenY))
# =============参数的配置 =====================
# Basic parameters
a = 0.1  # Diffusion constant
timesteps = 10000  # Number of time-steps to evolve system
image_interval = 1000  # Write frequency for png files
# Grid spacings
dx = 0.01
dy = 0.01
dx2 = dx ** 2
dy2 = dy ** 2
# For stability, this is the largest interval possible
# for the size of the time-step:
dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
# Boundary condition
Ttop = 100
Tbottom = 0
Tleft = 0
Tright = 30
# Initial guess of interior grid
Tguess = 0


def init_fields():
    # Set array size and set the interior value with Tguess
    field = np.empty((lenX, lenY))
    field.fill(Tguess)
    # Set Boundary condition
    field[(lenY - 1):, :] = Ttop
    field[:1, :] = Tbottom
    field[:, (lenX - 1):] = Tright
    field[:, :1] = Tleft

    print("size is ", field.size)
    print(field, "\n")
    return field


def write_field(field, step):
    # plt.gca().clear()
    plt.cla()
    plt.clf()

    # Configure the contour
    plt.title("Contour of Temperature")
    plt.contourf(X, Y, field, colorinterpolation, cmap=colourMap)
    # Set Colorbar
    plt.colorbar()
    plt.axis('on')
    plt.savefig('heat_Spark_{0:03d}.png'.format(step))
    # ============================================================


import pyspark
# 创建一个Spark上下文，这是连接到Spark集群的主要入口点
sc = pyspark.SparkContext.getOrCreate()
ps = (lenX, lenY, a, dt, dx, dy)


# 打印输入值，用于调试
def g(x):
    print(x)


def func(tup: tuple) -> list:
    """主要作用是把每个值打上行列index
    接收一个元组tup，其中包含一个数组和行索引
    作用是将每个值与其行列索引关联起来，返回一个包含行索引、列索引和值的列表。
    """
    Array, row_index = tup
    return [[row_index, col_index, value] for col_index, value in
            enumerate(Array)]


def merge_by_index(tup: tuple):
    """按row_index 把对应数据放入对应位置
    """
    row_index, data_list = tup
    res = [0] * len(data_list)
    for col_index, value in data_list:
        res[col_index] = value
    return (row_index, np.array(res, dtype=float))


# 内部扩散系数和四个方向上的扩散系数
internalSelf = 1.0 - 2.0 * a * dt / dx - 2.0 * a * dt / dy
diff4DirectionsDx = a * dt / dx
diff4DirectionsDy = a * dt / dy


# 接收一个包含行索引、列索引和值的列表list，计算该点与周围点之间的扩散值，返回一个包含扩散结果的列表
def diffuse(lst: list) -> list:
    # 包含矩阵的尺寸和热传导参数
    global ps
    # 用于存储结果的列表
    res = []
    # 检查是否为内部点，即不在边界上的点
    if lst[0] > ps[0] and lst[0] < ps[1] and lst[1] > ps[0] and lst[1] < ps[1]:
        # 对内部点，首先计算其自身的温度衰减值，并将其加入结果列表
        res.append([lst[0], lst[1], lst[2] * internalSelf])
        # 对四个边界条件进行检查，处理内部点扩散到其相邻点的情况，如果满足边界条件，则计算其扩散值
        if lst[0] - 1 == ps[0]:
            res.append([lst[0] + 1, lst[1], lst[2] * diff4DirectionsDx])
        elif lst[0] + 1 == ps[1]:
            res.append([lst[0] - 1, lst[1], lst[2] * diff4DirectionsDx])
        elif lst[1] - 1 == ps[0]:
            res.append([lst[0], lst[1] + 1, lst[2] * diff4DirectionsDy])
        elif lst[1] + 1 == ps[1]:
            res.append([lst[0], lst[1] - 1, lst[2] * diff4DirectionsDy])
        # 如果不在边界条件中，则计算该点向四个方向的扩散值，并将其加入结果列表
        else:
            res.append([lst[0] + 1, lst[1], lst[2] * diff4DirectionsDx])
            res.append([lst[0] - 1, lst[1], lst[2] * diff4DirectionsDx])
            res.append([lst[0], lst[1] + 1, lst[2] * diff4DirectionsDy])
            res.append([lst[0], lst[1] - 1, lst[2] * diff4DirectionsDy])
    # 对于边界点，首先将其自身的温度值加入结果列表
    else:
        res.append([lst[0], lst[1], lst[2]])
        # 对四个角上的点，不做扩散处理；对其他边界点，计算有效的扩散方向，并将扩散值加入结果列表
        if (lst[0], lst[1]) not in [(0, 0), (0, ps[1] - 1), (ps[0] - 1, 0), (ps[0] - 1, ps[1] - 1)]:
            if lst[0] == 0:
                res.append([lst[0] + 1, lst[1], lst[2] * diff4DirectionsDx])
            elif lst[0] == ps[0] - 1:
                res.append([lst[0] - 1, lst[1], lst[2] * diff4DirectionsDx])
            elif lst[1] == 0:
                res.append([lst[0], lst[1] + 1, lst[2] * diff4DirectionsDy])
            elif lst[1] == ps[1] - 1:
                res.append([lst[0], lst[1] - 1, lst[2] * diff4DirectionsDy])
    return res


# 负责单步时间演化
# 先为每个矩阵行元素分配索引，并通过flatMap函数调用func和diffuse函数进行扩散计算
# 然后通过reduceByKey聚合相同索引的值
# 最终将结果重新格式化为矩阵并返回
def evolve(matrixRDD: pyspark.RDD) -> pyspark.RDD:
    matrix = matrixRDD.zipWithIndex().flatMap(lambda x:
                                              func(x)).flatMap(lambda x: diffuse(x)) \
        .map(lambda x: ((x[0], x[1]), x[2])).reduceByKey(lambda x, y: x + y) \
        .map(lambda x: [x[0][0], x[0][1], x[1]]).map(lambda x: (x[0],
                                                                (x[1], x[2]))).groupByKey().mapValues(list).collect()
    return matrix


def main():
    # 初始化温度场mat
    mat = init_fields()
    # 保持初始状态
    write_field(mat, 0)
    # 将矩阵并行化为RDD
    rdd = sc.parallelize(mat)
    # rdd.foreach(g)
    # print()
    # 记录开始计算时间
    starting_time = timeit.default_timer()
    # 调用evolve函数进行时间演化，将结果重写并行化为RDD
    # 根据间隔条件保存当前状态的图像，最后打印总耗时并停止Spark上下文
    for m in range(5):
        # for m in range(1, timesteps + 1):
        # 调用演化函数，将matrix列表并行化为一个RDD
        matrix = evolve(rdd)
        # 将matrix列表并行化为一个RDD
        rdd6 = sc.parallelize(matrix)
        # 处理并排序rdd6
        rdd7 = np.asarray(rdd6.map(lambda
                                       x: merge_by_index(x)).sortBy(lambda x: x[0], ascending=True) \
                          .map(lambda x: x[1]).collect())
        # if m % image_interval == 0:
        if m % 2 == 0:
            write_field(rdd7, m)

        rdd = sc.parallelize(rdd7)
        # rdd.foreach(g)
        # print()
        # print("Iteration finished")

    print("Iteration finished. {} Seconds for Time difference: ".format(timeit.default_timer() - starting_time))
    sc.stop()


if __name__ == "__main__":
    main()

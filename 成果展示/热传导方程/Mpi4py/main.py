"""
https://github.com/csc-training/hpc-python/blob/master/mpi/heat
equation/solution/heat-p2p.py
"""

from __future__ import print_function
import numpy as np
import time
from mpi4py import MPI
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 初始参数的设置大部分和串行代码相同，修改了总步数和保存图像的步间隔，还修改了扩散系数
# Set the colormap
plt.rcParams['image.cmap'] = 'jet'
# Basic parameters
a = 0.5  # Diffusion constant
timesteps = 40000  # Number of time-steps to evolve system
image_interval = 600  # Write frequency for png files
# Grid spacings
dx = 0.01
dy = 0.01
dx2 = dx ** 2
dy2 = dy ** 2
# For stability, this is the largest interval possible
# for the size of the time-step:
dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
# 定义MPI相关的内容，用于并行计算
# 获取MPI通信子，代表所有参与计算的进程
comm = MPI.COMM_WORLD
# 获取当前进程的排名
rank = comm.Get_rank()
# 获取总的进程数量
size = comm.Get_size()

# 设置上下邻居进程
# 设置当前进程的上邻居进程ID为rank-1
up = rank - 1
# 如果上邻居进程ID小于0，则设置为MPI.PROC_NULL，表示没有上邻居
if up < 0:
    up = MPI.PROC_NULL
# 设置当前进程的下邻居进程ID为rank+1
down = rank + 1
# 如果下邻居进程ID超过总进程数量，则设置为MPI.PROC_NULL，表示没有下邻居
if down > size - 1:
    down = MPI.PROC_NULL


# 场演化函数，同串行代码
def evolve(u, u_previous, a, dt, dx2, dy2):
    """Explicit time evolution.
       u:            new temperature field
       u_previous:   previous field
       a:            diffusion constant
       dt:           time step
       dx2:          grid spacing squared, i.e. dx^2
       dy2:            -- "" --          , i.e. dy^2"""
    u[1:-1, 1:-1] = u_previous[1:-1, 1:-1] + a * dt * (
            (u_previous[2:, 1:-1] - 2 * u_previous[1:-1, 1:-1] +
             u_previous[:-2, 1:-1]) / dx2 +
            (u_previous[1:-1, 2:] - 2 * u_previous[1:-1, 1:-1] +
             u_previous[1:-1, :-2]) / dy2)
    u_previous[:] = u[:]


# 这个是原本代码的初始化场函数，但是苦于没有找到孔老师代码中的文件，于是新写了一个函数
# def init_fields(filename):
#     # Read the initial temperature field from file
#     field = np.loadtxt(filename)
#     field0 = field.copy()  # Array for field of previous time step
#     return field, field0

# 初始化场函数，同串行代码
def init_fields():
    # Set array size and set the interior value with Tguess
    field = np.empty((400, 400))
    field.fill(0)
    # Set Boundary condition
    field[(400 - 1):, :] = 100
    field[:1, :] = 0
    field[:, (400 - 1):] = 0
    field[:, :1] = 30
    print("size is ", field.size)
    print(field, "\n")
    field0 = field.copy()  # Array for field of previous time step
    return field, field0


# 保存图像的代码
def write_field(field, step):
    plt.title("Contour of Temperature")
    plt.gca().clear()
    plt.imshow(field)
    plt.axis('on')
    plt.savefig('heat_{0:03d}.png'.format(step))


# exchange函数用于在相邻的MPI进程之间交换边界数据，以便在每个时间步后更新边界条件
def exchange(field):
    # 取当前进程网格的倒数第二行作为发送缓冲区
    sbuf = field[-2, :]
    # 取当前进程网格的第一行作为接收缓冲区
    rbuf = field[0, :]
    # 将sbuf发送到下邻居进程，并从上邻居进程中接收数据到rbuf
    comm.Sendrecv(sbuf, dest=down, recvbuf=rbuf, source=up)
    # 取当前进程网格的第二行作为发送缓冲区
    sbuf = field[1, :]
    # 取当前进程的最后一行作为接收缓冲区
    rbuf = field[-1, :]
    # 将sbuf发送到上邻居进程，并从下邻居进程接收数据到rbuf
    comm.Sendrecv(sbuf, dest=up, recvbuf=rbuf, source=down)


# iterate函数执行整个时间演化过程
def iterate(field, local_field, local_field0, timesteps, image_interval):
    # 每次循环代表一个时间步
    for m in range(1, timesteps + 1):
        # 在相邻进程之间交换边界数据
        exchange(local_field0)
        # 根据扩散方程更新温度场
        evolve(local_field, local_field0, a, dt, dx2, dy2)
        # 每隔image_interval步保存一次图像
        if m % image_interval == 0:
            # 将所有进程的局部网格数据收集到根进程
            comm.Gather(local_field[1:-1, :], field, root=0)
            # 如果是根进程，则保存图像
            if rank == 0:
                write_field(field, m)


def main():
    # 在根进程上读取并分发初始温度场
    if rank == 0:
        field, field0 = init_fields()
        shape = field.shape
        dtype = field.dtype
        # 广播温度场的维度
        comm.bcast(shape, 0)
        # 广播温度场的数据类型
        comm.bcast(dtype, 0)
    else:
        # 其它进程接收广播的形状和数据类型
        field = None
        shape = comm.bcast(None, 0)
        dtype = comm.bcast(None, 0)
    # 检查温度场的函数可否被进程数整除
    if shape[0] % size:
        raise ValueError('Number of rows in the temperature field (' \
                         + str(shape[0]) + ') needs to be divisible by thenumber' \
                         + 'of MPI tasks(' + str(size) + ').')

    # 计算每个MPI任务需要处理的行数
    n = int(shape[0] / size)
    # 获取温度场的列数
    m = shape[1]
    # 创建一个缓冲区用于接收分发的数据
    buff = np.zeros((n, m), dtype)
    # 将温度场数据分发给各个进程，每个进程接收n行数据
    comm.Scatter(field, buff, 0)
    # 为每个进程创建包含两个幽灵行的局部温度场
    local_field = np.zeros((n + 2, m), dtype)
    # 将接收到的数据复制到局部温度场的非幽灵行
    local_field[1:-1, :] = buff
    # 创建用于前一步温度场的数组
    local_field0 = np.zeros_like(local_field)

    # 修正外边界的幽灵层，以处理非周期性边界
    if True:
        # 如果是第一个进程，将第一行设置为第二行的值
        if rank == 0:
            local_field[0, :] = local_field[1, :]
        # 如果是最后一个进程，将最后一行设置为倒数第二行的值
        if rank == size - 1:
            local_field[-1, :] = local_field[-2, :]

    # 初始化前一步的温度场为当前温度场
    local_field0[:] = local_field[:]
    # 绘制和保存初始温度场
    if rank == 0:
        write_field(field, 0)

    # 开始迭代，记录开始和结束时间
    t0 = time.time()
    iterate(field, local_field, local_field0, timesteps, image_interval)
    t1 = time.time()

    # 绘制和保存最终温度场
    # 将所有进程的局部温度场数据收集到根进程
    comm.Gather(local_field[1:-1, :], field, root=0)
    # 根进程保存最终温度场图像，打印总的运行时间
    if rank == 0:
        write_field(field, timesteps)
        print("Running time: {0}".format(t1 - t0))


if __name__ == '__main__':
    main()

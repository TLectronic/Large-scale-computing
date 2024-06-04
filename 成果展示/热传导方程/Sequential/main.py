'''
https://github.com/csc-training/hpc-python/blob/master/mpi/heat
equation/solution/heat-p2p.py
"D:\myCodes\HPCprojects\SourceCodes\parallel_python-master\mpi4py
heatequ2Dk.py"
'''

from __future__ import print_function
import numpy as np
import matplotlib
# 好像没有用到啊
import timeit

# 使用Agg后端，生成并保存图像文件
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置绘图参数
# plt.rcParams['image.cmap'] = 'BrBG'
# 设定颜色映射为‘jet’
plt.rcParams['image.cmap'] = 'jet'
# 设置图像分辨率为300 dpi
plt.figure(dpi=300)
# 设置颜色插值数量
colorinterpolation = 100
# 设置颜色映射
# colourMap = plt.cm.jet  # you can try: colourMap = plt.cm.coolwarm
colourMap = plt.cm.coolwarm
# Set Dimension 定义网格和基本参数
# 定义网格的尺寸
lenX = lenY = 400  # we set it rectangular
# Set meshgrid 生成网格点
# np.arange(0, lenX) 生成一个0到lenX-1的数组
# np.meshgrid接受两个一维数组，返回两个二维矩阵，分别表示所有网格点的x和y坐标
X, Y = np.meshgrid(np.arange(0, lenX), np.arange(0, lenY))
# 基本参数
# 扩散常数
a = 0.1
# 时间步数
timesteps = 10000  # Number of time-steps to evolve system
# 每1000步保存一次图像
image_interval = 1000  # Write frequency for png files
# dx和dy分别代表x与y方向上的网格间距，值都为0.01
dx = 0.01
dy = 0.01
# dx2和dy2分别代表x和y方向上的网格间距的平方
dx2 = dx ** 2
dy2 = dy ** 2
# 时间步长，保证稳定性
# 时间步长太小会增加计算时间，时间步长太大会导致数值解不稳定
# 时间步长的计算公式来源于时间步长dt的稳定性条件，它确保了显式方法中温度场的数值解是稳定的
dt = dx2 * dy2 / (2 * a * (dx2 + dy2))

# 定义边界条件和初始场
# 定义边界条件，分别是顶部、底部、左边和右边的温度
Ttop = 100
Tbottom = 0
Tleft = 0
Tright = 30
# 定义内部网格点的初始猜测温度
Tguess = 0


# 初始化场函数
def init_fields():
    # 创建一个空数组
    field = np.empty((lenX, lenY))
    # 将内部网格点初始化为Tguess
    field.fill(Tguess)
    # 设置顶部、底部、左右的边界
    # 选择数组最后一行的所有元素，将它们设置为顶部的温度值
    field[(lenY - 1):, :] = Ttop
    # 选择数组第一行的所有元素，将它们设置为底部的温度值
    field[:1, :] = Tbottom
    # 选择数组中最后一列的所有元素，将它们设置为右侧的温度值
    field[:, (lenX - 1):] = Tright
    # 选择数组中第一列的所有元素，将它们设置为左侧的温度值
    field[:, :1] = Tleft
    # 打印场数组的大小，为400*400=160000
    print("size is ", field.size)
    # 打印场数据
    print(field, "\n")
    # 复制初始场作为前一步的场
    field0 = field.copy()
    # 返回初始场和场的副本
    return field, field0


# 演化函数，根据扩散方程更新温度场
# 通过显示时间步进法来模拟二维热传导方程的演化，根据当前温度场和扩散方程计算下一时间步的温度场
# 计算每个内部网格点的新温度，并更新前一步的温度场
def evolve(u, u_previous, a, dt, dx2, dy2):
    """Explicit time evolution.
       u:            new temperature field 新的温度场
       u_previous:   previous field 前一步的温度场
       a:            diffusion constant 扩散常数
       dt:           time step 时间步长
       dx2:          grid spacing squared, i.e. dx^2
       dy2:            -- "" --          , i.e. dy^2"""
    # 更新温度场u中的内点（不包括边界点）
    # 计算公式是扩散方程的离散化形式，对x和y方向的导数项进行离散化
    # u[1:-1, 1:-1]表示从第二行选到倒数第二行，从第二列选到倒数第二列
    u[1:-1, 1:-1] = u_previous[1:-1, 1:-1] + a * dt * (
            (u_previous[2:, 1:-1] - 2 * u_previous[1:-1, 1:-1] +
             u_previous[:-2, 1:-1]) / dx2 +
            (u_previous[1:-1, 2:] - 2 * u_previous[1:-1, 1:-1] +
             u_previous[1:-1, :-2]) / dy2)
    # 更新前一个时间步的温度场
    u_previous[:] = u[:]


# 写场函数，将当前温度场绘制成等高线图，保存为PNG文件
def write_field(field, step):
    # plt.gca().clear()
    # 清除当前轴
    plt.cla()
    # 清除当前图形
    plt.clf()

    # Configure the contour
    # 设置图像标题
    plt.title("Contour of Temperature")
    # 使用contourf函数绘制温度场的等高线图
    plt.contourf(X, Y, field, colorinterpolation, cmap=colourMap)
    # 添加颜色条
    plt.colorbar()
    # 打开坐标轴
    plt.axis('on')
    # 保存图像
    plt.savefig('heat_Seq_{0:03d}.png'.format(step))


# 主函数
def main():
    # 初始场和其副本
    field, field0 = init_fields()
    # 将初始场的情况保存在图片中
    write_field(field, 0)
    # 进入循环，根据已有参数不断演化温度场
    for m in range(1, timesteps + 1):
        evolve(field, field0, a, dt, dx2, dy2)
        # 检查当前时间步m是否为image_interval的倍数，如果是则保存图像
        if m % image_interval == 0:
            write_field(field, m)


if __name__ == '__main__':
    main()

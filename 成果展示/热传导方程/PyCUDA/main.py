from __future__ import print_function
import numpy as np
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set the colormap
# plt.rcParams['image.cmap'] = 'BrBG'
plt.rcParams['image.cmap'] = 'jet'  # you can try: colourMap = plt.cm.coolwarm
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Set colour interpolation and colour map
colorinterpolation = 100
colourMap = plt.cm.jet  # you can try: colourMap = plt.cm.coolwarm

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule

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
# Set Dimension and delta
lenX = lenY = 400  # we set it rectangular
delta = 1
# Boundary condition
Ttop = 100
Tbottom = 0
Tleft = 0
Tright = 30
# Initial guess of interior grid
Tguess = 0
# Set meshgrid
X, Y = np.meshgrid(np.arange(0, lenX), np.arange(0, lenY))


def write_field(field, step):
    plt.gca().clear()
    # plt.clf()
    plt.figure(dpi=300)
    # Configure the contour
    plt.title("Contour of Temperature")
    plt.contourf(X, Y, field, colorinterpolation, cmap=colourMap)
    # Set Colorbar
    plt.colorbar()
    plt.axis('on')
    plt.savefig('heat_PyCUDA_{0:03d}.png'.format(step))


# 将用于CUDA内核的参数转换为适合CUDA使用的数据类型
a = np.float32(a)
lenX = np.int32(lenX)
lenY = np.int32(lenY)
dx = np.float32(dx)
dy = np.float32(dy)
dx2 = np.float32(dx2)
dy2 = np.float32(dy2)
dt = np.float32(dt)
timesteps = np.int32(timesteps)
image_interval = np.int32(image_interval)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# evolve_kernel 鏉ヨ嚜 D:\myCodes\HPCprojects\heat-equation-main\cuda\core_cuda.cu
# CUDA内核函数定义，用于更新温度场中的每个点
# 通过五点差分格式计算每个内部网格点在下一时间步的温度
# 保持固定的边界条件，因此最外层的网格点不被更新
ker = SourceModule('''
/* Update the temperature values using five-point stencil */
/*_global_修饰符表示这是一个在GPU上运行的CUDA内核函数*/
__global__ void evolve_kernelCUDA(float *currdata, float *prevdata, float a, float dt, int nx, int ny,float dx2, float dy2)
 {
    /* Determine the temperature field at next time step
     * As we have fixed boundary conditions, the outermost gridpoints
     * are not updated. */
     /*每个CUDA线程负责计算一个网格点的温度更新*/
    int ind, iRight, iLeft, jUp, jDown;
    // CUDA threads are arranged in column major order; thus j index from x, i from y
    //blockIdx和threadIdx是CUDA提供的内建变量，用于标识当前线程在网格中的位置
    //计算i和j确定当前线程负责的网格点的位置
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;        
    if (i > 0 && j > 0 && i < nx-1 && j < ny-1) {
        //当前网格点的线性索引，用于访问数组
        ind = i * ny + j;
        //当前网格点相邻点的索引
        iRight = (i+1)  * (ny) + j;
        iLeft = (i - 1) * (ny ) + j;
        jUp = i * (ny) + j + 1;
        jDown = i * (ny) + j - 1;
        //计算新温度值并存储在currdata中
        currdata[ind] = prevdata[ind] + a * dt *
          ((prevdata[iRight] -2.0 * prevdata[ind] + prevdata[iLeft]) / dx2 +
          (prevdata[jUp] - 2.0 * prevdata[ind] + prevdata[jDown]) / dy2);
    }
 }
 ''')
# 获取CUDA内核函数
evolve_kernel = ker.get_function('evolve_kernelCUDA')


# 浠ｇ爜鏉ヨ嚜 D:\myCodes\HPCprojects\heat-equation-main\cuda\core_cuda.cu

# void evolve(field *curr, field *prev, double a, double dt)
# {
# int nx, ny;
# double dx2, dy2;
# nx = prev->nx;
# ny = prev->ny;
# dx2 = prev->dx * prev->dx;
# dy2 = prev->dy * prev->dy;
# /* CUDA thread settings */
# const int blocksize = 16;  //!< CUDA thread block dimension
# dim3 dimBlock(blocksize, blocksize);
# // CUDA threads are arranged in column major order; thus make ny x nx grid
# dim3 dimGrid((ny + 2 + blocksize - 1) / blocksize,
# (nx + 2 + blocksize - 1) / blocksize);
# evolve_kernel<<<dimGrid, dimBlock>>>(curr->devdata, prev->devdata, a, dt, nx, ny, dx2, dy2);
# cudaDeviceSynchronize();
# }
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def main():
    # 初始化场
    field = np.empty((lenX, lenY))
    field.fill(Tguess)
    # 设置边界条件
    field[(lenY - 1):, :] = Ttop
    field[:1, :] = Tbottom
    field[:, (lenX - 1):] = Tright
    field[:, :1] = Tleft

    # gpuarray.to_gpu(field, allocator=None)
    # field = gpuarray.to_gpu(np.random.randn(lenX, lenX).astype(np.float32))
    # field_gpuArr = gpuarray.to_gpu(field.astype(np.float32))

    print("field's size is ", field.size)
    print(field, "\n")
    # 复制初始场的副本作为上一步场
    field0 = field.copy()  # Array for field of previous time step
    print("field0's size is ", field0.size)
    print(field0, "\n")
    # 将初始场保存为图像
    write_field(field, 0)
    # 将字段数组转换为float32类型，以便在GPU上进行计算
    field = field.astype(np.float32)
    field0 = field0.astype(np.float32)
    # 设置CUDA线程块和网格维度
    blocksize = 32  # !< CUDA thread block dimension
    # 线程块为32*32的二维块
    dimBlock = (blocksize, blocksize, 1)
    # CUDA threads are arranged in column major order; thus make ny x nx grid
    # dimGrid = ((lenY + 2 + blocksize - 1) // blocksize,
    # (lenX + 2 + blocksize - 1) // blocksize,
    # 1)
    # 网格的维度，根据字段数组大小和线程块的大小计算得出
    dimGrid = (int(lenX / blocksize + (0 if lenY % blocksize == 0 else 1)),
               int(lenY / blocksize + (0 if lenX % blocksize == 0 else 1)),
               1)
    print(dimBlock)
    print(dimGrid)
    print()

    # 在GPU上分配与field和field0相同大小的内存
    field_gpu = cuda.mem_alloc(field.nbytes)
    field0_gpu = cuda.mem_alloc(field0.nbytes)
    # 开始迭代
    t0 = time.time()
    for m in range(1, timesteps + 1):
        # 将field和field0复制到设备内存
        cuda.memcpy_htod(field_gpu, field)
        cuda.memcpy_htod(field0_gpu, field0)

        evolve_kernel(field_gpu, field0_gpu, a, dt, lenX, lenY, dx2, dy2, block=dimBlock, grid=dimGrid)
        # cuda.cudaDeviceSynchronize()
        # cuda.Context.synchronize()
        # 每隔image_interval步将结果从GPU复制回主机内存，保存当前温度场

        if (m % image_interval == 0):
            # 把结果从GPU复制回主机
            cuda.memcpy_dtoh(field, field_gpu)
            write_field(field, m);
            # print(field, "\n")

        cuda.memcpy_dtoh(field, field_gpu)
        cuda.memcpy_dtoh(field0, field0_gpu)
        # 将当前字段和前一个时间步的字段交换，以便下一次迭代使用
        tmp = field
        field = field0
        field0 = tmp

    t1 = time.time()
    print("Running time: {0}".format(t1 - t0))
    # 释放GPU上分配的内存
    field_gpu.free()
    field0_gpu.free()


if __name__ == '__main__':
    main()

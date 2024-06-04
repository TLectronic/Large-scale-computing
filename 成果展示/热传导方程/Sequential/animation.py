import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

SAVE_ANIMATION = True  # False True
# Set the colormap
# plt.rcParams['image.cmap'] = 'BrBG'
plt.rcParams['image.cmap'] = 'jet'  # you can try: colourMap = plt.cm.coolwarm
# plt.figure(dpi=300)

# Set colour interpolation and colour map
colorinterpolation = 100
colourMap = plt.cm.jet  # you can try: colourMap = plt.cm.coolwarm
# Basic parameters
a = 0.1  # Diffusion constant  #例如 Thermal diffusivity of steel, mm2.s-1
timesteps = 100000  # Number of time-steps to evolve system
image_interval = 1000  # Write frequency for png files
# Grid spacings
dx = 0.01
dy = 0.01
dx2 = dx ** 2
dy2 = dy ** 2
# For stability, this is the largest interval possible
# for the size of the time-step:
dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
# Set Dimension
lenX = lenY = 4000  # we set it rectangular
# Set meshgrid
X, Y = np.meshgrid(np.arange(0, lenX), np.arange(0, lenY))
# Boundary condition
Ttop = 100
Tbottom = 0
Tleft = 0
Tright = 30
# Initial guess of interior grid
Tguess = 30


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

    field0 = field.copy()  # Array for field of previous time step
    return field, field0

u, u0 = init_fields()
def evolve(u, u0, a, dt, dx2, dy2):
# Propagate with forward-difference in time, central-difference in space
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + a * dt * (
            (u0[2:, 1:-1] - 2 * u0[1:-1, 1:-1] +
             u0[:-2, 1:-1]) / dx2 +
            (u0[1:-1, 2:] - 2 * u0[1:-1, 1:-1] +
             u0[1:-1, :-2]) / dy2)
    u0[:] = u[:]
    # u0 = u.copy()
    return u, u0

fig, ax = plt.subplots()
# Configure the contour
plt.title("Contour of Temperature")
plt.contourf(X, Y, u0, colorinterpolation, cmap=colourMap)
# Set Colorbar
plt.colorbar()
plt.axis('on')


def animate(i):
    """Set the data for the ith iteration of the animation."""
    global u0, u, a, dt, dx2, dy2

    # plt.gca().clear()
    plt.cla()
    plt.clf()
    plt.title('{:.1f} ms'.format(i * dt * 1000))
    plt.contourf(X, Y, u0, colorinterpolation, cmap=colourMap)
    # Set Colorbar
    plt.colorbar()
    plt.axis('on')

    u, u0 = evolve(u, u0, a, dt, dx2, dy2)

    return u0, u


if SAVE_ANIMATION:

    anim = animation.FuncAnimation(fig, animate, frames=timesteps,
                               repeat=False, interval=50)
    # anim.save('us.gif', writer='imagemagick', fps=5)
    # anim.save("heat_equation_solution.gif", writer='imagemagick', fps=10)
    anim.save("heat_equation_solution.mp4", writer='ffmpeg', fps=10)
else:
    anim = animation.FuncAnimation(fig, animate, frames=timesteps)
    plt.show()
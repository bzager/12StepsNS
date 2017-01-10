# step7.py
# 2D Diffusion
# u_t = nu*(u_xx + u_yy)
# 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

nx = 31
ny = 31
length = 2.0
time = 0.5
dx = length / (nx-1)
dy = length / (ny-1)
nu = 0.05
sigma = 0.25
dt = sigma*dx*dy/nu
nt = int(time/dt)

x = np.linspace(0,length,nx)
y = np.linspace(0,length,ny)
X,Y = np.meshgrid(x,y)

u = np.ones((ny,nx))
un = np.ones((ny,nx))

# Initial conditions
u[int(0.5/dy):int(1/dy + 1),int(0.5/dx):int(1/dx + 1)] = 2


fig = plt.figure()

def diffuse(time):
	u[int(0.5/dy):int(1/dy + 1),int(0.5/dx):int(1/dx + 1)] = 2 
	nt = int(time/dt)
	for n in range(nt+1):
		print n,
		un = u.copy()
		u[1:-1,1:-1] = (un[1:-1,1:-1] + 
						nu*dt/dx**2*
						(un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2]) + 
						nu*dt/dy**2*
						(un[2:,1:-1] - 2*un[1:-1,1:-1] + un[0:-2,1:-1]))
		u[0,:] = 1
		u[-1,:] = 1
		u[:,0] = 1
		u[:,-1] = 1

		fig.clear()
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(X,Y,u[:],rstride=1,cstride=1,cmap=cm.viridis,
					linewidth=0,antialiased=True)
		ax.set_zlim(1,2.5)
		ax.set_xlabel('$x$')
		ax.set_ylabel('$y$')
		plt.pause(0.0001)
	plt.show()


diffuse(1)


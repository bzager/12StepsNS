# step6.py
# 2D Nonlinear convection
# u_t + u*u_x + v*u_y = 0
# v_t + u*v_x + v*v_y = 0
# Initial conditions: u,v = 2 for x,y in (0.5,1)x(0.5,1)
#					  u,v = 1 everywhere else 
# Boundary conditions: u=1, v=1 for x=0,2 y=0,2 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

nx = 51
ny = 51
length = 2.0
time = 1.0
dx = length / (nx-1)
dy = length / (ny-1)
c = 1.0
sigma = 0.2
dt = sigma*dx
nt = int(time/dt)

x = np.linspace(0,length,nx)
y = np.linspace(0,length,ny)
X,Y = np.meshgrid(x,y)

u = np.ones((ny,nx))
v = np.ones((ny,nx))
un = np.ones((ny,nx))
vn = np.ones((ny,nx))

# Initial conditions
u[int(0.5/dy):int(1/dy + 1),int(0.5/dx):int(1/dx + 1)] = 2
v[int(0.5/dy):int(1/dy + 1),int(0.5/dx):int(1/dx + 1)] = 2


fig = plt.figure()
#axes = subplots(nrows=1,ncols=2)

for n in range(nt+1):
	un = u.copy()
	vn = v.copy()

	u[1:,1:] = (un[1:,1:] - 
				(un[1:,1:]*c*dt/dx*(un[1:,1:]-un[1:,:-1])) - 
				(vn[1:,1:]*c*dt/dy*(un[1:,1:]-un[:-1,1:])))

	v[1:,1:] = (vn[1:,1:] - 
				(un[1:,1:]*c*dt/dx*(vn[1:,1:]-vn[1:,:-1])) - 
				(vn[1:,1:]*c*dt/dy*(vn[1:,1:]-vn[:-1,1:])))

	u[0,:] = 1
	u[-1,:] = 1
	u[:,0] = 1
	u[:,-1] = 1
	v[0,:] = 1
	v[-1,:] = 1
	v[:,0] = 1
	v[:,-1] = 1

	if (n%5 == 0):
		fig.clear()
		ax = fig.gca(projection="3d")
		surf = ax.plot_surface(X,Y,u[:],rstride=1,cstride=1,linewidth=0)#,cmap=cm.viridis)
		plt.title(str(n*dt))
		plt.pause(0.0001)

plt.show()





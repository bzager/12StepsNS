# step8.py
# 2D Burgers' equation
# u_t + u*u_x + v*u_y = nu*(u_xx + u_yy)
# v_t + u*v_x + v*v_y = nu*(v_xx + v_yy)
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

nx = 25
ny = 25
length = 2.0
time = 2.0
dx = length / (nx-1)
dy = length / (ny-1)
c = 1.0
sigma = 0.0009
nu = 0.01
dt = sigma*dx*dy/nu
nt = int(time/dt)

x = np.linspace(0,length,nx)
y = np.linspace(0,length,ny)
X,Y = np.meshgrid(x,y)

u = np.ones((ny,nx))
v = np.ones((ny,nx))
un = np.ones((ny,nx))
vn = np.ones((ny,nx))
comb = np.ones((ny,nx))

# Initial conditions
u[int(0.5/dy):int(1/dy + 1),int(0.5/dx):int(1/dx + 1)] = 2
v[int(0.5/dy):int(1/dy + 1),int(0.5/dx):int(1/dx + 1)] = 2


fig = plt.figure()

# Solve
for n in range(nt+1):
	un = u.copy()
	vn = v.copy()

	u[1:-1,1:-1] = (un[1:-1,1:-1] -
                     dt/dx*un[1:-1,1:-1] * 
                     (un[1:-1,1:-1] - un[1:-1,0:-2]) - 
                     dt/dy*vn[1:-1,1:-1] * 
                     (un[1:-1,1:-1] - un[0:-2,1:-1]) + 
                     nu * dt / dx**2 * 
                     (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2]) + 
                     nu * dt / dy**2 * 
                     (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[0:-2,1:-1]))

	v[1:-1,1:-1] = (vn[1:-1,1:-1] - 
                     dt/dx*un[1:-1,1:-1] *
                     (vn[1:-1,1:-1] - vn[1:-1,0:-2]) -
                     dt/dy*vn[1:-1,1:-1] * 
                    (vn[1:-1,1:-1] - vn[0:-2,1:-1]) + 
                     nu*dt/dx**2 * 
                     (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,0:-2]) +
                     nu*dt/dy**2 *
                     (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[0:-2,1:-1]))
	u[0,:] = 1
	u[-1,:] = 1
	u[:,0] = 1
	u[:,-1] = 1
	v[0,:] = 1
	v[-1,:] = 1
	v[:,0] = 1
	v[:,-1] = 1

	if (n%20 == 0):
		fig.clear()
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(X,Y,u[:],rstride=1,cstride=1,cmap=cm.viridis,
						linewidth=0,antialiased=False)
		ax.set_zlim(1,2.5)
		ax.set_xlabel('$x$')
		ax.set_ylabel('$y$')
		plt.pause(0.0001)
	
plt.show()




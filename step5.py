# step5.py
# 2D Linear Convection
# Initial conditions: u_0(x) = 2, 0.5<=x<=1, u_0(x) = 1, everywhere else
# Boundary conditions: u(x,y) = 1, x=0,2 y=0,2
# (i.e. u = 1 on boundary of 2x2 square)
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.colors import LightSource
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
un = np.ones((ny,nx))

# Initial conditions
u[int(0.5/dy):int(1/dy + 1),int(0.5/dx):int(1/dx + 1)] = 2

# Create figure
fig = plt.figure()

# Nested loop implementation
"""
for n in range(nt):
	print n,
	un = u.copy()
	row,col = u.shape
	for j in range(1,row):
		for i in range(1,col):
			u[j,i] = (un[j,i] - (c*dt/dx*(un[j,i]-un[j,i-1])) - 
								(c*dt/dy*(un[j,i]-un[j-1,i])))
			u[0,:] = 1
			u[-1,:] = 1
			u[:,0] = 1
			u[:,-1] = 1
	
	if (n%2 == 0):
		fig.clear()
		ax = fig.gca(projection="3d")
		surf = ax.plot_surface(X,Y,u[:],rstride=1,cstride=1,linewidth=0,cmap=cm.viridis)
		plt.title(str(n*dt))
		plt.pause(0.0001)

plt.show()
"""

# Vectorized implementation

for n in range(nt+1):
	un = u.copy()
	u[1:,1:] = (un[1:,1:] - (c*dt/dx*(un[1:,1:]-un[1:,:-1])) - 
							(c*dt/dy*(un[1:,1:]-un[:-1,1:])))
	u[0,:] = 1
	u[-1,:] = 1
	u[:,0] = 1
	u[:,-1] = 1

	if (n%5 == 0):
		fig.clear()
		ax = fig.gca(projection="3d")
		surf = ax.plot_surface(X,Y,u[:],rstride=1,cstride=1,linewidth=0)#,cmap=cm.viridis)
		plt.title(str(n*dt))
		plt.pause(0.0001)

plt.show()



# step10.py
# 2D Poisson equation
# p_xx + p_yy = b
# 
# 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 

def plot2D(x,y,p):
	fig = plt.figure()
	ax = fig.gca(projection="3d")
	X,Y = np.meshgrid(x,y)
	surf = ax.plot_surface(X,Y,p[:],rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=False)
	ax.set_xlim(0, 2)
	ax.set_ylim(0, 1)
	ax.view_init(30, 225)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	plt.show()


def poisson2D(x,y,p,b,dx,dy,nx,ny,nt):

	for it in range(nt):
		pd = p.copy()
		p[1:-1,1:-1] = (((pd[1:-1,2:] + pd[1:-1,0:-2])*dy**2 + 
						(pd[2:,1:-1] + pd[0:-2,1:-1])*dx**2 -
						b[1:-1,1:-1] * dx**2 * dy**2) / 
						(2*(dx**2 + dy**2)))
		p[:,0] = 0
		p[ny-1,:] = 0
		p[:,0] = 0 
		p[:,nx-1] = 0

	return p



nx = 50
ny = 50
nt  = 100
xmin = 0
xmax = 2.0
ymin = 0
ymax = 1.0

dx = (xmax-xmin)/(nx-1)
dy = (ymax-ymin)/(ny-1)

p = np.zeros((ny,nx))
pd = np.zeros((ny,nx))
b = np.zeros((ny,nx))
x = np.linspace(xmin,xmax,nx)
y = np.linspace(ymin,ymax,ny)

b[int(ny/4),int(nx/4)] = 100
b[int(3*ny/4),int(3*nx/4)] = -100

p = poisson2D(x,y,p,b,dx,dy,nx,ny,nt)
plot2D(x,y,p)


	
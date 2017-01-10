# step9.py
# 2D Laplace equation
# p_xx + p_yy = 0
# 5-point difference operator
# Initial conditions: p(x,y) = 0 for all x,y
# Boundary conditions: p(0,y)=0, p(2,y)=y, p_y(x,0)=p_y(x,1)=0
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


def laplace2D(x,y,p,dx,dy,tol):
	l1norm = 1
	pn = np.empty_like(p)
	iter = 0

	while l1norm > tol:
		iter += 1
		pn = p.copy()
		p[1:-1,1:-1] = ((dy**2 * (pn[1:-1,2:] + pn[1:-1,0:-2]) + 
							dx**2 * (pn[2:,1:-1] + pn[0:-2,1:-1])) / 
							(2*(dx**2 + dy**2)))
		p[:,0] = 0  # p(0,y) = 0
		p[:,-1] = y  # p(2,y) = y
		p[0,:] = p[1,:]  # p_y(x,0) = 0
		p[-1,:] = p[-2,:]  # p_y(x,1) = 0
		l1norm = (np.sum(np.abs(p[:]) - np.abs(pn[:])) /
				 np.sum(np.abs(pn[:])))
	return p


nx = 31
ny = 31
length = 2.0
c = 1.0
dx = length/ (nx-1)
dy = length/ (ny-1)
tol = 1e-4
# Initial conditions
p = np.zeros((ny,nx))

x = np.linspace(0,length,nx)
y = np.linspace(0,length/2,ny)

# Boundary conditions
p[:,0] = 0  		# p(0,y) = 0
p[:,-1] = y 		# p(length,y) = y
p[0,:] = p[1,:] 	# p_y(x,0) = 0
p[-1,:] = p[-2,:] 	# p_y(x,length/2) = 0


p = laplace2D(x,y,p,dx,dy,tol)
plot2D(x,y,p)



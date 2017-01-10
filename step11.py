# step11.py
# Cavity flow with Navier-Stokes
# 
# u_t +u*u_x + v*u_y = (-1/rho)*p_x + nu*(u_xx + u_yy)
# v_t +u*v_x + v*v_y = (-1/rho)*p_y + nu*(v_xx + v_yy)
# p_xx + p_yy = -rho*((u_x)^2 + (v_y)^2 + 2*u_y*v_x)
# Periodic boundary conditions

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def build_up_b(b,rho,dt,u,v,dx,dy):

	b[1:-1,1:-1] = (rho*(1/dt*
				((u[1:-1,2:]-u[1:-1,0:-2])/(2*dx) + 
				(v[2:,1:-1]-v[0:-2,1:-1])/(2*dy)) - 
				((u[1:-1,2:]-u[1:-1,0:-2])/(2*dx))**2 - 
				2*((u[2:,1:-1]-u[0:-2,1:-1])/(2*dy) * 
				(v[1:-1,2:]-v[1:-1,0:-2])/(2*dx)) - 
				((v[2:,1:-1]-v[0:-2,1:-1])/(2*dy))**2))
	return b

def pressure_poisson(p,dx,dy,b,nit):
	pn = np.empty_like(p)
	pn = p.copy()

	for q in range(nit):
		pn = p.copy()
		p[1:-1,1:-1] = (((pn[1:-1,2:] + pn[1:-1,0:-2])*dy**2 + 
					(pn[2:,1:-1] + pn[0:-2,1:-1])*dx**2) / 
					(2*(dx**2 + dy**2)) - dx**2 * dy**2 / 
					(2*(dx**2 + dy**2)) * b[1:-1,1:-1])
		# Wall BC
		p[:,-1] = p[:,-2]
		p[0,:] = p[1,:]
		p[:,0] =  p[:,1]
		p[-1,:] = 0

	return p

def cavity_flow(nt,nit,rho,nu,x,y,u,v,p,dt,dx,dy):
	un = np.empty_like(u)
	vn = np.empty_like(v)
	b = np.zeros((ny,nx))

	fig = plt.figure()
	X,Y = np.meshgrid(x,y)

	for n in range(nt):
		un = u.copy()
		vn = v.copy()

		b = build_up_b(b,rho,dt,u,v,dx,dy)
		p = pressure_poisson(p,dx,dy,b,nit)

		u[1:-1,1:-1] = (un[1:-1,1:-1]-un[1:-1,1:-1] * dt/dx * 
					(un[1:-1,1:-1]-un[1:-1,0:-2]) - 
					vn[1:-1,1:-1] * dt/dy * (un[1:-1,1:-1]-un[0:-2,1:-1]) -
					dt/(2*rho*dx) * (p[1:-1,2:]-p[1:-1,0:-2]) + 
					nu*(dt/dx**2 * (un[1:-1,2:] - 2*un[1:-1,1:-1] + 
					un[1:-1,0:-2]) + dt/dy**2 * (un[2:,1:-1] - 
					2*un[1:-1,1:-1] + un[0:-2,1:-1])))

		v[1:-1,1:-1] = (vn[1:-1,1:-1]-un[1:-1,1:-1] * dt/dx * 
					(vn[1:-1,1:-1]-vn[1:-1,0:-2]) - 
					vn[1:-1,1:-1] * dt/dy * (vn[1:-1,1:-1]-vn[0:-2,1:-1]) -
					dt/(2*rho*dx) * (p[2:,1:-1]-p[0:-2,1:-1]) + 
					nu*(dt/dx**2 * (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + 
					vn[1:-1,0:-2]) + dt/dy**2 * (vn[2:,1:-1] - 
					2*vn[1:-1,1:-1] + vn[0:-2,1:-1])))
		# Wall BC u,v = 0 at y=0,2
		u[0,:] = 0
		u[:,0] = 0
		u[:,-1] = 0
		u[-1,:] = 1
		v[0,:] = 0
		v[-1,:]= 0
		v[:,0] = 0
		v[:,-1] = 0

		if (n % 100 == 0):
			fig.clear()
			plt.contourf(X,Y,p,alpha=0.5,cmap=cm.viridis)
			plt.colorbar()
			#plt.contour(X,Y,p,cmap=cm.viridis)
			vfield = plt.quiver(X[::2,::2],Y[::2,::2],u[::2,::2],v[::2,::2])
			plt.xlabel('$X$')
			plt.ylabel('$Y$')
			plt.title(n*dt)
			plt.pause(0.0001)
	
	plt.show() 

	return u,v,p

def plot2D():

	#fig = plt.figure()
	"""
	plt.contourf(X,Y,p,alpha=0.5,cmap=cm.viridis)
	plt.colorbar()
	plt.contour(X,Y,p,cmap=cm.viridis)
	plt.quiver(X[::2,::2], Y[::2,::2],u[::2,::2],v[::2,::2])
	plt.xlabel('$X$')
	plt.ylabel('$Y$')
	plt.show() 
	"""
	"""
	ax = fig.gca(projection="3d")
	X,Y = np.meshgrid(x,y)
	surf = ax.plot_surface(X,Y,p[:],rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=False)
	ax.set_xlim(0, 2)
	ax.set_ylim(0, 1)
	ax.view_init(30, 225)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	plt.show()
	"""

nx = 41
ny = 41
nit = 50
time = 2
length = 2.0
c = 1.0
dx = length / (nx-1)
dy = length / (ny-1)
x = np.linspace(0,length,nx)
y = np.linspace(0,length,ny)
X,Y = np.meshgrid(x,y)

# Physical parameters
rho = 1.0
nu = 0.1
dt = 0.002
nt = int(time/dt)

# Initial conditions
u = np.zeros((ny,nx))
v = np.zeros((ny,nx))
p = np.zeros((ny,nx))
b = np.zeros((ny,nx))

u,v,p = cavity_flow(nt,nit,rho,nu,x,y,u,v,p,dt,dx,dy)



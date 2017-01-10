# step12.py
# Channel flow with Navier-Stokes
#
# Driving force F in x-direction
# u_t +u*u_x + v*u_y = (-1/rho)*p_x + nu*(u_xx + u_yy) + F
# v_t +u*v_x + v*v_y = (-1/rho)*p_y + nu*(v_xx + v_yy)
# p_xx + p_yy = -rho*((u_x)^2 + (v_y)^2 + 2*u_y*v_x)
# 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def build_up_b(rho,dt,dx,dy,u,v):
	b = np.zeros_like(u)
	b[1:-1,1:-1] = (rho*(1/dt*((u[1:-1,2:]-u[1:-1,0:-2])/(2*dx) + 
					(v[2:,1:-1]-v[0:-2,1:-1])/(2*dy)) - 
					((u[1:-1,2:]-u[1:-1,0:-2])/(2*dx))**2 - 
					2*((u[2:,1:-1]-u[0:-2,1:-1])/(2*dy) * 
					(v[1:-1,2:]-v[1:-1,0:-2])/(2*dx)) - 
					((v[2:,1:-1]-v[0:-2,1:-1])/(2*dy))**2))
	# Periodic BC pressure at x = 2
	b[1:-1,-1] = (rho*(1/dt*((u[1:-1,0]-u[1:-1,-2])/(2*dx) + 
					(v[2:,-1]-v[0:-2,-1])/(2*dy)) - 
					((u[1:-1,0]-u[1:-1,-2])/(2*dx))**2 - 
					2*((u[2:,-1]-u[0:-2,-1])/(2*dy) * 
					(v[1:-1,0]-v[1:-1,-2])/(2*dx)) - 
					((v[2:,-1]-v[0:-2,-1])/(2*dy))**2))
	# Periodic BC pressure at x = 0
	b[1:-1,0] = (rho*(1/dt*((u[1:-1,1]-u[1:-1,-1])/(2*dx) + 
					(v[2:,0]-v[0:-2,0])/(2*dy)) - 
					((u[1:-1,1]-u[1:-1,-1])/(2*dx))**2 - 
					2*((u[2:,0]-u[0:-2,0])/(2*dy) * 
					(v[1:-1,1]-v[1:-1,-1])/(2*dx)) - 
					((v[2:,0]-v[0:-2,0])/(2*dy))**2))
	return b


def pressure_poisson_periodic(p,dx,dy,nit):
	pn = np.empty_like(p)

	for q in range(nit):
		pn = p.copy()
		p[1:-1,1:-1] = (((pn[1:-1,2:] + pn[1:-1,0:-2])*dy**2 + 
					(pn[2:,1:-1] + pn[0:-2,1:-1])*dx**2) / 
					(2*(dx**2 + dy**2)) - dx**2 * dy**2 / 
					(2*(dx**2 + dy**2)) * b[1:-1,1:-1])

		# Periodic BC pressure at x = 2
		p[1:-1,-1] = (((pn[1:-1,0] + pn[1:-1,-2])*dy**2 + 
					(pn[2:,-1] + pn[0:-2,-1])*dx**2) / 
					(2*(dx**2 + dy**2)) - dx**2 * dy**2 / 
					(2*(dx**2 + dy**2)) * b[1:-1,-1])

		# Periodic BC pressure at x = 0
		p[1:-1,0] = (((pn[1:-1,1] + pn[1:-1,-1])*dy**2 + 
					(pn[2:,0] + pn[0:-2,0])*dx**2) / 
					(2*(dx**2 + dy**2)) - dx**2 * dy**2 / 
					(2*(dx**2 + dy**2)) * b[1:-1,0])

		# Wall BC
		p[-1,:] = p[-2,:]
		p[0,:] =  p[1,:]

	return p


def channel_flow(p,nit,rho,dt,dx,dy,x,y,u,v,tol):
	udiff = 1
	count = 0
	fig = plt.figure()
	X,Y = np.meshgrid(x,y)

	while udiff > tol:
		un = u.copy()
		vn = v.copy()

		b = build_up_b(rho,dt,dx,dy,u,v)
		p = pressure_poisson_periodic(p,dx,dy,nit)

		u[1:-1,1:-1] = (un[1:-1,1:-1]-un[1:-1,1:-1] * dt/dx * 
					(un[1:-1,1:-1]-un[1:-1,0:-2]) - 
					vn[1:-1,1:-1] * dt/dy * (un[1:-1,1:-1]-un[0:-2,1:-1]) -
					dt/(2*rho*dx) * (p[1:-1,2:]-p[1:-1,0:-2]) + 
					nu*(dt/dx**2 * (un[1:-1,2:] - 2*un[1:-1,1:-1] + 
					un[1:-1,0:-2]) + dt/dy**2 * (un[2:,1:-1] - 
					2*un[1:-1,1:-1] + un[0:-2,1:-1])) + F*dt)

		v[1:-1,1:-1] = (vn[1:-1,1:-1]-un[1:-1,1:-1] * dt/dx * 
					(vn[1:-1,1:-1]-vn[1:-1,0:-2]) - 
					vn[1:-1,1:-1] * dt/dy * (vn[1:-1,1:-1]-vn[0:-2,1:-1]) -
					dt/(2*rho*dx) * (p[2:,1:-1]-p[0:-2,1:-1]) + 
					nu*(dt/dx**2 * (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + 
					vn[1:-1,0:-2]) + dt/dy**2 * (vn[2:,1:-1] - 
					2*vn[1:-1,1:-1] + vn[0:-2,1:-1])) + F*dt)

		# Periodic BC u @ x = 2
		u[1:-1,-1] = (un[1:-1,-1]-un[1:-1,-1] * dt/dx * 
					(un[1:-1,-1]-un[1:-1,-2]) - 
					vn[1:-1,-1] * dt/dy * (un[1:-1,-1]-un[0:-2,-1]) -
					dt/(2*rho*dx) * (p[1:-1,0]-p[1:-1,-2]) + 
					nu*(dt/dx**2 * (un[1:-1,0] - 2*un[1:-1,-1] + 
					un[1:-1,-2]) + dt/dy**2 * (un[2:,-1] - 
					2*un[1:-1,-1] + un[0:-2,-1])) + F*dt)
		# Periodic BC u @ x = 0
		u[1:-1,0] = (un[1:-1,0]-un[1:-1,0] * dt/dx * 
					(un[1:-1,0]-un[1:-1,-1]) - 
					vn[1:-1,0] * dt/dy * (un[1:-1,0]-un[0:-2,0]) -
					dt/(2*rho*dx) * (p[1:-1,1]-p[1:-1,-1]) + 
					nu*(dt/dx**2 * (un[1:-1,1] - 2*un[1:-1,0] + 
					un[1:-1,-1]) + dt/dy**2 * (un[2:,0] - 
					2*un[1:-1,0] + un[0:-2,0])) + F*dt)
		# Periodic BC v @ x = 2
		v[1:-1,-1] = (vn[1:-1,-1]-un[1:-1,-1] * dt/dx * 
					(vn[1:-1,-1]-vn[1:-1,-2]) - vn[1:-1,-1] * dt/dy * 
					(vn[1:-1,-1]-vn[0:-2,-1]) - dt/(2*rho*dy) * 
					(p[2:,-1]-p[0:-2,-1]) + nu*(dt/dx**2 * 
					(vn[1:-1,0] - 2*vn[1:-1,-1] + vn[1:-1,-2]) + dt/dy**2 * 
					(vn[2:,-1] - 2*vn[1:-1,-1] + vn[0:-2,-1])))
		# Periodic BC v @ x = 0
		v[1:-1,0] = (vn[1:-1,0]-un[1:-1,0] * dt/dx * 
					(vn[1:-1,0]-vn[1:-1,-1]) - vn[1:-1,0] * dt/dy * 
					(vn[1:-1,0]-vn[0:-2,0]) - dt/(2*rho*dy) * 
					(p[2:,0]-p[0:-2,0]) + nu*(dt/dx**2 * 
					(vn[1:-1,1] - 2*vn[1:-1,0] + vn[1:-1,-1]) + dt/dy**2 * 
					(vn[2:,0] - 2*vn[1:-1,0] + vn[0:-2,0])))

		# Wall BC u,v = 0 at y=0,2
		u[0,:] = 0
		u[-1,:] = 0
		v[0,:] = 0
		v[-1,:]= 0

		udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
		count += 1
		
		if (count % 2 == 0):
			fig.clear()
			vfield = plt.quiver(X[::3,::3],Y[::3,::3],u[::3,::3],v[::3,::3])
			plt.xlabel('$X$')
			plt.ylabel('$Y$')
			#plt.title()
			plt.pause(0.0001)
	plt.show() 

	return u,v

nx = 41
ny = 41
nit = 50
length = 2.0
time = 2.0
c = 1.0
dx = length / (nx-1)
dy = length / (ny-1)
x = np.linspace(0,length,nx)
y = np.linspace(0,length,ny)
X,Y = np.meshgrid(x,y)
tol = 0.00001

# Physical parameters
rho = 1.0
nu = 0.1
F = 1.0
dt = 0.01
nt = int(time/dt)

# Initial conditions
u = np.zeros((ny,nx))
un = np.zeros((ny,nx))
v = np.zeros((ny,nx))
vn = np.zeros((ny,nx))
p = np.ones((ny,nx))
pn = np.ones((ny,nx))
b = np.zeros((ny,nx))

u,v,p = channel_flow(p,nit,rho,dt,dx,dy,x,y,u,v,tol)



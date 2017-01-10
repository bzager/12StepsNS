# bugers.py
# Comparison of finite differce schemes for 1D Burgers' equation
# u_t = -u*u_x, OR
# u_t = -(u^2 / 2)_x
# BC: u(x,0) = 1 at 0<=x<=2 and u(x,0) = 0 at 2<=x<=4
# Lax-Friedrichs, Lax-Wendroff, MacCormack, Beam & Warming Implicit,
# using substitution E = 0.5*u^2, u_t = -E_x

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import animation
from JSAnimation import IPython_display 

nx = 101
time = 10.0
length = 4.0
sigma = 1.0
dx = length / nx
dt = sigma*dx
nt = int(time / dt)

def u_ic(nx):
	#u = np.ones(nx)
	#u[(nx-1)/3:2*(nx-1)/3]=0
	x = np.linspace(0,length,nx)
	u = np.cos(2*np.pi*x)

	return u

utoE = lambda u: (u/2)**2
utoA = lambda u: u**2

u = u_ic(nx)

fig = plt.figure()
ax = plt.axes(xlim=(0,length),ylim=(-1.0,2.0))
line, = ax.plot([],[],lw=2)


# ANIMATIONS
def animate(data):
	x = np.linspace(0,length,nx)
	y = data
	line.set_data(x,y)
	return line,

# Lax-Friedrichs
# explicit 1st-order, FTCS
# u(i,n+1) = 0.5*(u(i+1,n)+u(i-1,n)) - 0.5*(dt/dx)*(E(i+1,n)-E(i-1,n))
def laxFriedrichs(u,nt,dt,dx):
	un = np.zeros((nt,len(u)))
	un[:,:] = u.copy()
	for i in range(1,nt):
		E = utoE(u)
		un[i,1:-1] = .5*(u[2:]+u[:-2]) - dt/(2*dx)*(E[2:]-E[:-2])
		un[i,0] = 1
		u = un[i].copy()

	return un

# Lax-Wendroff
# substitute E_t = -A*E_x, and u_t = -E_x
#  
def laxWendroff(u,nt,dt,dx):
	un = np.zeros((nt,len(u)))
	un[:] = u.copy()

	for i in range(1,nt):
		E = utoE(u)
		un[i,1:-1] = u[1:-1] - dt/(2*dx) * (E[2:]-E[:-2]) + dt**2/(4*dx**2) *\
		((u[2:]+u[1:-1])*(E[2:]-E[1:-1]) -\
		(u[1:-1]+u[:-2])*(E[1:-1]-E[:-2]))
		un[i,0]=1
		u = un[i].copy()

	return un

# MacCormack 
# 'predictor-corrector' method
# 
def macCormack(u,nt,dt,dx):
	un = np.zeros((nt,len(u)))
	ustar = np.empty_like(u)
	un[:] = u.copy()
	ustar = u.copy()
	for i in range(1,nt):
		E = utoE(u)
		ustar[:-1] = u[:-1] - dt/dx * (E[1:]-E[:-1])
		Estar = utoE(ustar)
		un[i,1:] = .5 * (u[1:]+ustar[1:] - dt/dx * (Estar[1:] - Estar[:-1]))
		u = un[i].copy()

	return un

# Beam & Warming, implicit
# results in tridiagonal system
#
def beamWarming(u,nt,dt,dx):
    ##Tridiagonal setup##
    a = np.zeros_like(u)
    b = np.ones_like(u)
    c = np.zeros_like(u)
    d = np.zeros_like(u)

    un = np.zeros((nt,len(u)))
    un[:]=u.copy()
    for n in range(1,nt): 
    	u[0] = 1
    	E = utoE(u)
        au = utoA(u)
    
        a[0] = -dt/(4*dx)*u[0]
        a[1:] = -dt/(4*dx)*u[0:-1]
        a[-1] = -dt/(4*dx)*u[-1]

        c[:-1] = dt/(4*dx)*u[1:]
        d[1:-1] = u[1:-1]-.5*dt/dx*(E[2:]-E[0:-2])+dt/(4*dx)*(au[2:]-au[:-2])

        ###subtract a[0]*LHS B.C to 'fix' thomas algorithm
        d[0] = u[0] - .5*dt/dx*(E[1]-E[0])+dt/(4*dx)*(au[1]-au[0]) - a[0]
        ab = np.matrix([c,b,a])
        u = linalg.solve_banded((1,1), ab, d)
        u[0]=1
        un[n] = u.copy()        
    return un

# 
def dampit(u,eps,dt,dx):
	d = u[2]-.5*dt/dx*(u[3]**2/2-u[1]**2/2)+dt/(4*dx)*(u[3]**2-u[1]**2)\
		-eps*(u[4]-4*u[3]+6*u[2]-4*u[1]+u[0])
	return d

def beamWarmingDamp(u,nt,dt,dx):
	##Tridiagonal setup##
	a = np.zeros_like(u)
	b = np.ones_like(u)
	c = np.zeros_like(u)
	d = np.zeros_like(u)

	un = np.zeros((nt,len(u)))
	un[:] = u.copy()
	eps = .01

	for n in range(1,nt): 
		u[0] = 1
		E = utoE(u)
		au = utoA(u)

		a[0] = -dt/(4*dx)*u[0]
		a[1:] = -dt/(4*dx)*u[0:-1]
		a[-1] = -dt/(4*dx)*u[-1]

		c[:-1] = dt/(4*dx)*u[1:]

		###Calculate the damping factor for MOST of our u_vector
		d[2:-2] = u[2:-2]-.5*dt/dx*(E[3:-1]-E[1:-3])+dt/(4*dx)\
			*(au[3:-1]-au[1:-3])\
			-eps*(u[4:]-4*u[3:-1]+6*u[2:-2]-4*u[1:-3]+u[:-4])
		###Calculate the damping factor for d[0] and d[1]
		damp = np.concatenate((np.ones(2), u[:3]))
		d[0] = dampit(damp,eps,dt,dx)
		damp = np.concatenate((np.ones(1), u[:4]))
		d[1] = dampit(damp,eps,dt,dx)

		###subtract a[0]*LHS B.C to 'fix' thomas algorithm
		d[0] = d[0] - u[0] * a[0]

		ab = np.matrix([c,b,a])
		u = linalg.solve_banded((1,1), ab, d)
		u[0]=1
		un[n] = u.copy()
	return un


# Lax-Friedrichs Test
un = laxFriedrichs(u,nt,dt,dx)
plt.title("Lax-Friedrichs Method, " + "CFL=" + str(sigma))

# Lax-Wendroff Test
#un = laxWendroff(u,nt,dt,dx)
#plt.title("Lax-Wendroff Method, " + "CFL=" + str(sigma))

# MacCormack Test
#un = macCormack(u,nt,dt,dx)
#plt.title("MacCormack Method, " + "CFL=" + str(sigma))

# Beam-Warming
#un = beamWarming(u,nt,dt,dx)
#plt.title("Beam-Warming, " + "CFL=" + str(sigma))

# Beam-Warming Damped
#un = beamWarmingDamp(u,nt,dt,dx)
#plt.title("Beam-Warming Damped, " + "CFL=" + str(sigma))


anim = animation.FuncAnimation(fig,animate,frames=un,interval=30)
plt.show()



# step4.py
# 1D Burgers' Equation
# u_t + u*u_x = v*u_xx
# boundary conditions: u(0) = u(2pi)
# initial conditions: u = -(2v/phi)*phi_x + 4
# where phi = exp(-(x-4t)^2/4v(t+1)) + exp(-(x-4t-2pi)^2/4v(t+1))


import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
#from sympy import init_printing
#init_printing(use_latex=True)
from sympy.utilities.lambdify import lambdify

# Initial Conditions
x,v,t = sp.symbols('x v t')
phi = (sp.exp(-(x - 4*t)**2 / (4*v*(t+1))) + 
	   sp.exp(-(x - 4*t - 2*np.pi)**2 / (4*v*(t+1))))
phi_x = phi.diff(x)

u = -2*v*(phi_x/phi) + 4
ufunc = lambdify((t,x,v),u)


nx = 101
time = 7
length = 2*np.pi
dx = length / (nx-1)
v = 0.1
dt = dx*v
nt = int(time/dt)

x = np.linspace(0,length,nx)
un = np.empty(nx)
t = 0

u = np.asarray([ufunc(t,x0,v) for x0 in x])


fig = plt.figure(figsize=(11,7),dpi=100)


for n in range(nt):
	un = u.copy()
	for i in range(1,nx-1):
		u[i] = un[i] - un[i]*dt/dx*(un[i]-un[i-1]) + v*dt/dx**2 * (un[i+1] - 2*un[i] + un[i-1])

	u[0] = un[0] - un[0]*dt/dx*(un[0]-un[-2]) + v*dt/dx**2 * (un[1] - 2*un[0] + un[-2])
	u[-1] = un[-1] - un[-1]*dt/dx*(un[-1]-un[-2]) + v*dt/dx**2 * (un[0] - 2*un[-1] + un[-2])

	if (n%10 == 0):
	    fig.clear()
	    plt.plot(x,u,lw=2)
	    plt.axis([0,2*np.pi,0,10])
	    plt.title(str(n*dt))
	    plt.pause(0.0001)


plt.show()



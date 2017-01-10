# step1.py
# Step 1 in "12 steps to Navier Stokes"
# 
# 1D linear convection, u_t + cu_x = 0, with u(x,0) = u_0(x)
# exact solution: u(x,t) = u_0(x-ct)
# forward diff. appx. for time derivative:
# backward diff. appx. for space derivative: 
#

import numpy as np
import matplotlib.pyplot as plt
import time, sys

nx = 41 		# space discretization
dx = float(2) / (nx-1)
x = np.linspace(0,2,nx)
nt = 25			# time discretization
dt = 0.025
c = 1 			# speed

# initial conditions, u_0=2 for 0.5<=x<=1, u_0=1 everywhere else
u = np.ones(nx)
u[int(0.5/dx):int(1/dx + 1)] = 2

#plt.plot(x, u)
#plt.show()


un = np.ones(nx)

# NOT AN EFFICIENT IMPLEMENTATION
for n in range(nt):
	un = u.copy()
	for i in range(1, nx):
		u[i] = un[i] - c * dt/dx * (un[i]-un[i-1])

plt.plot(x,u)
plt.show()
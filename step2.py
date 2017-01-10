# step2.py
# Step 2 in "12 steps to Navier Stokes"
# 1D Nonlinear convection
# u_t + u*u_x = 0
# 

import numpy as np
import matplotlib.pyplot as plt

length = 5.0
time = 2.0 # time in seconds
nx = 200
dx = length/(nx-1)
dt = 0.001
nt = int(time/dt)

x = np.linspace(0,length,nx)

u = np.ones(nx)
u[int(0.5/dx):int(1/dx + 1)] = 2

un = np.ones(nx)

fig = plt.figure()
plt.ion()

for n in range(nt):
	un = u.copy()
	for i in range(nx):
		u[i] = un[i] - un[i] * dt/dx * (un[i]-un[i-1])

	if (n%20 == 0):
		fig.clear()
		plt.plot(x,u)
		plt.axis([0,length,1,2])
		plt.pause(0.0001)

plt.show()


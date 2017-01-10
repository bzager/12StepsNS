# step3.py
# 1D Diffusion Equation, u_t = v*u_xx
# u_xx = (u_i+1 - 2u_i + u_i-1) / h^2


import numpy as np
import matplotlib.pyplot as plt

nx = 100
length = 2.4
time = 1 # secondss
dx = length / (nx-1)
x = np.linspace(0,length,nx)
v = 0.6
sigma = 0.2
dt = sigma * dx**2 / v
nt = int(time / dt)

# intial conditions
u = np.ones(nx)
u[int(0.5/dx):int(1/dx + 1)] = 2

un = np.ones(nx)

fig = plt.figure()
plt.ion()

for n in range(nt):
	un = u.copy()
	for i in range(1,nx-1):
		u[i] = un[i] + v*dt/dx**2 * (un[i+1]-2*un[i]+un[i-1])

	if (n%50 == 0):
		fig.clear()
		plt.plot(x,u)
		plt.axis([0,length,1,2])
		plt.title(str(n*dt))
		plt.pause(0.0001)

plt.show()


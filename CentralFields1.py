# MATH10222, a central fields example: exercise 37
# Rather than a conservation of energy approach, we instead compute
# the solution of Newton's second law directly
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import math
# graph range
xLeft = -1.5
xRight = 1.5
yBottom = -1.5
yTop = 1.5

# paramters
d = 1.0                   # initial radius of P from O
gamma = 1.0               # bigger gamma means more attractive force field
HO = math.sqrt(gamma)     # angular momentum constant

# time stepper for direct solution of Newton's 2nd law
dt = 0.005
tMax = 0.95

# force function, vector force is m*F(f)*r-hat-vector
def F(r):
    return -gamma/r**3

# Newton's 2nd law (in polar form) written as THREE first order equations
# d/dt [r,rdot,theta] = [rdot,F(r)+HO/r^3,HO/r^2]
# d/dt(theta) = HO/r^2
# systemState contains : [r,rdot,theta] where rdot=dr/dt
def NewtonSecond( systemState, t):
    derivState = np.zeros_like(systemState)
    derivState[0] = systemState[1]       # d/dt(r) = rdot
    derivState[1] = F(systemState[0]) + HO/systemState[0]**3  # d/dt(rdot) = F(r) + HO/r^3
    derivState[2] = HO/systemState[0]**2 # d/dt(theta) = HO/r^2 (obviously trivial to integrate)
    return derivState

# time points used in solving Newton's second law
t = np.arange(0.0,tMax,dt)

# initial condition: state[0]=r=d, state[1]=rdot=-sqrt(gamma)/d, state[2]=theta=0
state = np.array([d,-math.sqrt(gamma)/d,0.0])
soln = integrate.odeint(NewtonSecond, state, t)

# matplotlib for output
fig = plt.figure(figsize=(5,5))
ax = plt.axes()
particlePoint, = ax.plot([], [], 'o', color='r', markersize=8, zorder=20, label="particle")


plt.xlim(xLeft,xRight)
plt.ylim(yBottom,yTop)
plt.xlabel("position, x = r*cos(theta)")
plt.ylabel("position, y = r*sin(theta)")
plt.title("Exercise 37")

plt.plot(0.0,0.0, 'o') # origin of the force field at r=0

# construct the (X,Y) path from polar solution
X=[]
Y=[]
print(len(t))
for i in range(0,len(t)):
    X.append(soln[i,0]*math.cos(soln[i,2]))
    Y.append(soln[i,0]*math.sin(soln[i,2]))

# plot the path
plt.plot(X,Y,'-',color='k', linewidth=0.5, label="path")
plt.legend(loc='lower right', frameon=False)

# animate the position of the particle in the well function
def animate(i):
    # P is at x = r*cos(theta) and y = r*sin(theta)
    particlePoint.set_data(X[i],Y[i])
    
     
ani = animation.FuncAnimation(fig, animate, np.arange(1,len(t)), interval=10)
#ani.save("CentralFields.mp4", fps=25, dpi=100)
plt.show()

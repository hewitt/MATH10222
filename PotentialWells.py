# MATH10222, a potential well animation example: exercise 25
# Rather than a conservation of energy approach, we instead compute
# the solution of Newton's second law directly for given initial
# conditions of x=4, v=dx/dt=0.
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
# graph range
xLeft = 0.5
xRight = 6.0
yBottom = 3.0
yTop = 6.0
# time stepper for direct solution of Newton's 2nd law
dt = 0.05
tMax = 6.9
# mass of the particle
m = 1.0

# number of x points in the graph
N = 101

# potential function
def V(x):
    return 4/x+x

# force function, F(x)=-V'(x)
def F(x):
    return 4/(x*x)-1

# Newton's 2nd law written as TWO first order equations
# d/dt [x,v] = [v,F(x)/m]
# systemState contains the position and velocity: [x,v] where v=dx/dt
def NewtonSecond( systemState, t):
    derivState = np.zeros_like(systemState)
    derivState[0] = systemState[1]       # d/dt(x) = v
    derivState[1] = F(systemState[0])/m  # d/dt(v) = acceleration = F(x)/m
    return derivState

# x points used to plot V(x)
x = np.linspace(xLeft,xRight,N)
# time points used in solving Newton's second law
t = np.arange(0.0,tMax,dt)

# initial condition: state[0]=x=4, state[1]=v=0
state = np.array([4.0,0.0])
soln = integrate.odeint(NewtonSecond, state, t)

# matplotlib for output
fig = plt.figure(figsize=(5,5))
ax = plt.axes()
particlePoint, = ax.plot([], [], 'o', color='r', markersize=12, zorder=20)

plt.xlim(xLeft,xRight)
plt.ylim(yBottom,yTop)
plt.xlabel("position, x")
plt.title("Exercise 25")

# plot the potential well
plt.plot(x, V(x), label="potential function V(x)")
# plot the total energy (should be a straight line!)
plt.plot(soln[:,0], 0.5*m*soln[:,1]*soln[:,1]+V(soln[:,0]), label="total energy E")
plt.legend(loc='lower right', frameon=False)

# animate the position of the particle in the well function
def animate(i):
     particlePoint.set_data(soln[i,0],V(soln[i,0]))
     
ani = animation.FuncAnimation(fig, animate, np.arange(1,len(t)), interval=20)
ani.save("PotentialWells.mp4", fps=25)
plt.show()

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

def Bot(state,t, v=0.1, omega = 0.0):
  x = state[0]
  y = state[1]
  theta = state[2]

  # variables

  ddt = [v*math.cos(theta), v*math.sin(theta), omega]
  return ddt

state0 = [0.0, 0.0, 0]
t = np.linspace(0.0, 100, 200)

state = odeint(Bot, state0, np.linspace(0,(np.pi/4) * 10, 100), args=(0, 0.1))
state = np.concatenate((state,odeint(Bot, state[-1], t, args=(0.1,0))))
# state.append(odeint(Bot, state[-1], t, args=(0,0.1)))
# state = np.concatenate((state, odeint(Bot, state[-1], t, args=(0.1, 0))))
# state.append(odeint(Bot, state[-1], t, args=(0.1, 0)))

# print(state[:][0])
plt.plot(state[:,0], state[:,1])
# plt.plot(t, state)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('TurtleBot')
# plt.legend(('$x$ (m)', '$y$', '$\theta$'))  
plt.show()
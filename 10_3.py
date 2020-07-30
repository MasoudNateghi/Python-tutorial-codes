import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,10,500)
f = np.exp(-x/10)*np.sin(np.pi * x)
g = x * np.exp(-x/3)
plt.figure()
plt.subplot(2,1,1)
plt.xlabel('time')
plt.ylabel('f(x)')
plt.plot(x,f)
plt.grid()
plt.subplot(2,1,2)
plt.xlabel('time')
plt.ylabel('g(x)')
plt.plot(x,g)
plt.grid()
plt.show()

theta = np.linspace(0,2 * np.pi, 100)
r1 = 0.8 + np.cos(theta)
r2 = 1   + np.cos(theta)
r3 = 1.2 + np.cos(theta)
x1 = r1 * np.cos(theta)
x2 = r2 * np.cos(theta)
x3 = r3 * np.cos(theta)
y1 = r1 * np.sin(theta)
y2 = r2 * np.sin(theta)
y3 = r3 * np.sin(theta)
plt.figure()
plt.legend(['r = 0.8','r = 1.2'])
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.plot(x3,y3)
plt.legend(['r = 0.8','r = 1','r = 1.2'])
plt.grid()
plt.show()
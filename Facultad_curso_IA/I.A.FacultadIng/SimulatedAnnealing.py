import numpy as np
from matplotlib import pyplot as plt

def h(x):
    if x < -1 or x > 1:
        y = 0
    else:
        y = (np.cos(50 * x) + np.sin(20 * x))
    return y

hv = np.vectorize(h) #Vectorize the function
X = np.linspace(-1,2,num = 1000)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#Comparison of the greedy algorithm between the annealing
def simple_greedy_serch(func, start = 0,N = 100):
    x = start
    history = []
    for i in range(N):
        history.append(x)
        u = 0.001
        xleft, xright = x -u, x+u
        yleft, yright = func(xleft), func(xright)
        if yleft > yright:
            x = xleft
        else:
            x = xright
    return x, history

x0, history = simple_greedy_serch(hv, start = -0.02, N = 100)
plt.plot(X,hv(X))
plt.scatter(x0,h(x0), marker = 'x', s = 100)
plt.plot(history, hv(history))
plt.show()

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#Annealing Algorithm
def SA(search_space, func, T):
    scale = np.sqrt(T)
    start = np.random.choice(search_space)
    x = start * 1
    cur = func(x)
    history = [x]
    temperatures = [T]
    for i in range(9000):
        prop = x + np.random.normal()* scale
        if (prop > 1 or prop < 0 or np.log(np.random.rand()) * T > func(prop) - cur):
            prop = x
        x = prop
        cur = func(x)
        T = 0.9 * T
        temperatures.append(T)
        history.append(x)
    return x, history, temperatures
X = np.linspace(-1,1, num = 1000)
x1, history, temperatures = SA(X, h, T = 4)

plt.plot(X, hv(X))
plt.scatter(x1, hv(x1), marker = 'x')
plt.plot(history, hv(history))
#plt.plot(temperatures, hv(temperatures), marker = '*')
plt.show()

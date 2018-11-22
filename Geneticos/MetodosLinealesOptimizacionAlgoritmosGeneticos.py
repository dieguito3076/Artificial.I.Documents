import random
import numpy as np

from sympy import *
x = Symbol('x')
y = Symbol('y')

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#Metodo de optimizacion de Newton
def funcion_fx(x):
    return x ** 4 + 6 *x **3 + 2 * x ** 2 - x
def primera_derivada_funcion_fx(x):
    return 4 * x **3 + 18 * x ** 2 + 4 * x - 1
def segunda_derivada_funcion_fx(x):
    return 12 * x ** 2 + 36 * x + 4
def Metodo_De_Newton(deriv1,deriv2):
    x1 = random.random() * 10
    while(deriv1(x1) != 0):
        x1 = x1 - deriv1(x1) / deriv2(x1)
    if(x1 >=0):
        print "Se ha encontrado punto maximo"
    elif(x1 <0):
        print "Se ha encontrado punto minimo"
    return x1

def graficar(funcion, punto):
    for i in range(-50,50):
        plt.plot(i,funcion(i), color="red", marker = "*")
    plt.plot(punto, funcion(punto),color = "green", marker = "o")
    plt.show()
punto_critico = Metodo_De_Newton(primera_derivada_funcion_fx, segunda_derivada_funcion_fx)
#graficar(funcion_fx, punto_critico)
print "El punto critico es: "+str(punto_critico)

 #**************************************************************************
#fig = plt.figure()
#ax = fig.add_subplot(111,projection = '3d')
#z =  np.random.random((1,150))
#x =  np.random.random((1,150))
#y =  np.random.random((1,150))
#ax.plot_wireframe(x,y,z)

#Metodo de optimizacion de ascenso de maxima pendiente
def funcion_en_z_y_x(x, y):
    return -(x+2) ** 2 - (y-1)**2
def primera_derivada_funcion_yxz(x, y):
    return [-2*(x+2), -2*(y-1)]
def Metodo_Optimizacion_ascensoPendiente(funcion_inicial, primera_derivada):
    x1 =  [random.random()*10, random.random()*10]
    evaluacion_gradiente = primera_derivada(x1[0], x1[1])
    to = solve(diff(-(x1[0] + x * evaluacion_gradiente[0] - 5)**2 -(x1[1] + x * evaluacion_gradiente[1]-3)**2),x)
    nuevo_punto = [x1[0] + (to[0] * evaluacion_gradiente[0]), x1[1] + (to[0] * evaluacion_gradiente[1])]
    evaluacion_gradiente = primera_derivada(nuevo_punto[0], nuevo_punto[1])
    return nuevo_punto

print Metodo_Optimizacion_ascensoPendiente(funcion_en_z_y_x,primera_derivada_funcion_yxz)

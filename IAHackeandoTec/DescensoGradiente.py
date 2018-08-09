#-*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import math
import scipy as sc #funciones cientificas

#optimización de funciones
func = lambda th: np.sin(1 / 2 * th[0] ** 2 - 1/4*th[1]**2 +3) * np.cos(2 * th[0] +1 - np.e ** th[1])#función anónima, th es un parámetro

#generar un vector con una secuencua de x, y. Y los vamos a evaluar con la función para despu´pes representar en gráfia.
resolucion = 100
_x = np.linspace(-2, 2, resolucion) #Genera 100 valores aleatorios entre el -2 y 2
_y = np.linspace(-2, 2, resolucion)

_Z = np.zeros((resolucion,  resolucion)) #crear la matriz con ceros primero
for ix, x in enumerate(_x):
  for iy, y in enumerate(_y):
    _Z[iy, ix] = func([x, y])

plt.contourf(_x, _y, _Z, 100) #éste método pasamos las variables x, y, z. El número cien nos sirve para aumentar la resolución de la gráfica.
#contourf nos da la parte sólida mientras que contour nos dá la superficie en mallado.
plt.colorbar() #añade un indicador de valores con sus respectivos colores
#plt.show()
#print func([5,3]) #mandándola a llamar

#Ahora vamos a generar un punto aleatorio sobre la supeficie
theta = np.random.rand(2) * 4 -2 #números del rango -2 a 2
_T = np.copy(theta)

#ahora vamos con las derivadas parciales, que es donde calculamos la pendiente en dicho punto para poder ir descendiendo.
h = 0.001
ratioAprendizaje = 0.001
#generando el punto aleatorio

plt.plot(theta[0], theta[1], "o", c = "white")

gradiente = np.zeros(2)

for _ in range(10000):
  for it, th in enumerate(theta):
    _T = np.copy(theta)
    _T[it] = _T[it] + h
    deriv = (func(_T) - func(theta)) / h #derivada parcial del primer vector, o sea la pendiente de movernos en el eje Z[i]
    gradiente[it] = deriv
  #ahora actualizamos parámetros
  theta = theta - ratioAprendizaje * gradiente
  #cada 100 iteraciones se va a dibujar un puntito para ver el recorrido que va a ir haciendo.
  if(_ %100 == 0):
    plt.plot(theta[0], theta[1], ".", c = "red")

plt.plot(theta[0], theta[1], "o", c = "green")

plt.show()  #Aqupi visualizaré todo el plano tridimensional con un punto aleatorio sobre dicha superficie

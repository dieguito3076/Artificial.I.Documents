# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

p = np.array([[1, 1, 1, 1, 1, 1, 0], #0
              [0, 1, 1, 0, 0, 0, 0], #1
              [1, 1, 0, 1, 1, 0, 1], #2
              [1, 1, 1, 1, 0, 0, 1], #3
              [0, 1, 1, 0, 0, 1, 1], #4
              [1, 0, 1, 1, 0, 1, 1], #5
              [1, 0, 1, 1, 1, 1, 1], #6
              [1, 1, 1, 0, 0, 0, 0], #7
              [1, 1, 1, 1, 1, 1, 1], #8
              [1, 1, 1, 1, 0, 1, 1]])#9

cero    = p[0]
uno     = p[1]
dos     = p[2]
tres    = p[3]
cuatro  = p[4]
cinco   = p[5]
seis    = p[6]
siete   = p[7]
ocho    = p[8]
nueve   = p[9]

unos = np.array([1,1,1,1,1,1,1,1,1,1])
p = p.transpose()
#La matriz z surge de hacer transpuesta la matriz P y añadirle una matriz al final
#de unos

z =np.array( [ [1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
               [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
               [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
               [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
               [1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
               [0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

q = 10 #Tiene que regresarme el valor de 10
tpar =  np.array([ 1, -1,  1, -1,  1,  -1,  1,  -1,  1,  -1])
tmay5 = np.array([-1,  -1, -1, -1, -1,  -1,  1, 1,  1,  1  ])
tprim = np.array([-1,  -1,  1,  1,  -1,  1,  -1,  1,  -1,  -1])
T = np.array([tpar, tmay5, tprim])
#Aprendizaje -WidrowHoff

R =  float(1)/float(q)  * np.dot(z, z.transpose())
H = float(1)/float(q)   * np.dot(z, T.transpose())
Xm = np.dot(linalg.inv(R), H)

#Obteniendo los pesos sinápticos
w1 = []
w2 = []
w3 = []

for i in range(len(Xm) - 1):
  w1.append(Xm[i][0])
  w2.append(Xm[i][1])
  w3.append(Xm[i][2])

w1 = np.array(w1)
w2 = np.array(w2)
w3 = np.array(w3)

#Obteniendo las polaridades

b1 = Xm[7][0]
b2 = Xm[7][1]
b3 = Xm[7][2]

def net_input(entrada, W, B):
	 return np.dot(entrada, W) + B

def definir(expresion):
  if expresion < 0:
    print "desactivada\n"
  else:
    print "activada\n"

entrada = ocho
print "El número: " + str(entrada)
print "Números pares:"
definir(net_input(entrada, w1, b1))
print "Numeros mayores a 5:"
definir(net_input(entrada, w2, b2))
print "Numeros primos:"
definir(net_input(entrada, w3, b3))

numeros = np.array([cero, uno, dos, tres, cuatro, cinco, seis, siete, ocho, nueve])

for i in range(len(numeros)):
    plt.plot(net_input(numeros[i], w3, b3), b3, "o", c = "red")
    plt.plot(net_input(numeros[i], w2, b2), b2, "o", c = "green")
    plt.plot(net_input(numeros[i], w1, b1), b1, "o", c = "blue")
plt.show()

print len(R)
print len(H)
print len(Xm)

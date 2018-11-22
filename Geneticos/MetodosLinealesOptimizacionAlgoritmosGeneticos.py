import random
from matplotlib import pyplot as plt

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
graficar(funcion_fx, punto_critico)
print punto_critico

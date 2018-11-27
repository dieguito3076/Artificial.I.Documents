import numpy as np
esquema = "1**"

def crear_poblacion(esquema):
  count = 0
  #Evaluacion de '*' en la cadena
  for i in range(0,len(esquema)):
    if(esquema[i] == "*"):
      count = count + 1
  #Creacion de la matriz de zeros
  s = (2 ** count)
  combinatoria = np.zeros(s)
  #Declarando matriz que tendra la poblacion de cadenas binarias
  poblacion = [esquema] * (s)

  #Meter los numeros binarios a la matriz
  for i in range(0,len(combinatoria)):
    combinatoria[i] = bin(i)[2:]
  len_cadenas = len(str(int(combinatoria[len(combinatoria)-1])))

  #Ajustar todos los numeros binarios para que tengan el mismo numero de caracteres

  for i in range(0,len(combinatoria)):
    numero = str(int(combinatoria[i]))
    zeros = int(len_cadenas - (len(str(int(combinatoria[i]))))) * '0'
    nuevo_numero = zeros + numero
    #Aqui se crea un elemento de la poblacion con su combinatoria de bits

    numero_final = ""
    posicion = 0
    for a in range(0,len(esquema)):
      if(poblacion[i][a] != '*'):
        numero_final = numero_final + poblacion[i][a]
      elif(poblacion[i][a] == '*'):
        numero_final = numero_final + nuevo_numero[posicion]
        posicion = posicion + 1
    poblacion[i] = numero_final
    nuevo_numero = ""
  return poblacion

print(crear_poblacion(esquema))





#*****************************************************************************

import numpy as np
import random
esquema = "1**********1"

def crear_poblacion(esquema):
  count = 0
  #Evaluacion de '*' en la cadena
  for i in range(0,len(esquema)):
    if(esquema[i] == "*"):
      count = count + 1
  #Creacion de la matriz de zeros
  s = (2 ** count)
  combinatoria = np.zeros(s)
  #Declarando matriz que tendra la poblacion de cadenas binarias
  poblacion = [esquema] * (s)

  #Meter los numeros binarios a la matriz
  for i in range(0,len(combinatoria)):
    combinatoria[i] = bin(i)[2:]
  len_cadenas = len(str(int(combinatoria[len(combinatoria)-1])))

  #Ajustar todos los numeros binarios para que tengan el mismo numero de caracteres

  for i in range(0,len(combinatoria)):
    numero = str(int(combinatoria[i]))
    zeros = int(len_cadenas - (len(str(int(combinatoria[i]))))) * '0'
    nuevo_numero = zeros + numero
    #Aqui se crea un elemento de la poblacion con su combinatoria de bits

    numero_final = ""
    posicion = 0
    for a in range(0,len(esquema)):
      if(poblacion[i][a] != '*'):
        numero_final = numero_final + poblacion[i][a]
      elif(poblacion[i][a] == '*'):
        numero_final = numero_final + nuevo_numero[posicion]
        posicion = posicion + 1
    poblacion[i] = numero_final
    nuevo_numero = ""
  return poblacion


def retornar_indice(elemento, matriz):
  indice = 0
  for i in range(0,len(matriz)):
    if(matriz[i]==elemento):
      return indice
    else:
      indice = indice + 1

poblacion = (crear_poblacion(esquema))
print(poblacion)
print("Tamano de combinatoria de esquema: "+str(len(poblacion)))
numero_de_muestra = int(random.random() * len(poblacion) / 4)
print("Tamano de poblacion inicial: " + str(numero_de_muestra))
print("Una posible solucion está en el indice: "+ str(retornar_indice('110100001111', poblacion)))

#Obteniendo la muestra con la que trabajara el algoritmo genetico
poblacion_Muestra = poblacion[:numero_de_muestra]
print(poblacion_Muestra)

#Definiendo matriz de hijos
matriz_hijos = {"A":["B","C"],"B":["A","E","D"],"C":["A","D","G"],"D":["B","C","I"],"E":["B","F"],"F":["L","K","J","I"],"G":["C","H"],"H":["G","I"],"I":["F","J","D","H"],"J":["F","K","I"],"K":["F","J","L"],"L":["F","K"]}

class Algoritmo_Genetico:
  def __init__(self, poblacion, letra_stop, mutation_rate, matriz_hijos):
    self.poblacion = poblacion
    self.letra_stop = letra_stop
    self.mutation_rate = mutation_rate
    self.matriz_hijos = matriz_hijos
    #Matriz de apoyo letras
    self.matriz_apoyo_letras = ["A","B","C","D","E","F","G","H","I","J","K","L"]
    #Numero de generacion
    self.numero_generacion = 0
    #Matriz que contendra las medidas fitness de cada elemento de la poblacion
    self.matriz_medidas_fit = np.zeros(len(self.poblacion))
    #Generaciones que quiero que haga el algoritmo
    self.total_generations = 250

  def funcion_fitness(self, elemento):
    nodo_padre = self.matriz_apoyo_letras[0] #El nodo padre inicial es A
    for i in range(1, len(elemento)):
      if(elemento[i] == 1):
        nodo_hijo = self.matriz_apoyo_letras[i]
        if(nodo_hijo in self.matriz_hijos[nodo_padre]):
          medida_fit = medida_fit + 1
          nodo_padre = self.matriz_apoyo_letras[i]
          if(nodo_padre == self.letra_stop):
            medida_fit = 10
    return medida_fit

  def retornar_indice(self,elemento, matriz):
    indice = 0
    for i in range(0,len(matriz)):
      if(matriz[i]==elemento):
        return indice
      else:
        indice = indice + 1

  def evolve_main(self):
    for i in range(0,self.total_generations):

      #Asignando todas las medidas fitness
      for u in range(0,len(self.poblacion)):
        self.matriz_medidas_fit[u] = self.funcion_fitness(poblacion[u])

      #Elitismo (En la posicion cero se guardará el mejor elemento de la generacion pasada)
      indice_elemento_max = self.retornar_indice(max(self.matriz_medidas_fit),self.matriz_medidas_fit)
      elemento_max_fitness = self.poblacion[indice_elemento_max]
      self.poblacion[0] = elemento_max_fitness
      #Main del algoritmo genetico
      for a in range(1,len(poblacion)):
        padre, madre = self.bernouli_selection()
        hijo = self.cruzamiento(padre, madre)
        hijo = self.mutacion(hijo)
        self.poblacion[i] = hijo

#-*- coding: utf-8 -*-
import numpy as np
import random
from matplotlib import pyplot as plt
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
#*****************************************************************************

esquema = "1**********1"

poblacion = (crear_poblacion(esquema))
print("Tamano de combinatoria de esquema: "+str(len(poblacion)))
poblacion_muestra = poblacion[300:400]
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
    #Población que tendrá a los hijos provisionales
    self.poblacion_con_hijos = poblacion
    #Generaciones que quiero que haga el algoritmo
    self.total_generations = 50
  def funcion_fitness(self, hijo):
    elemento = hijo
    nodo_padre = self.matriz_apoyo_letras[0] #El nodo padre inicial es A
    medida_fit = 0
    i = 1
    while(i < len(elemento)):
      if(elemento[i] ==  '1'):
        nodo_hijo = self.matriz_apoyo_letras[i]
        if(nodo_hijo in self.matriz_hijos[nodo_padre]):
          medida_fit = medida_fit + 1
          nodo_padre = self.matriz_apoyo_letras[i]
        else:
          i = len(elemento) + 1
      i = i + 1
    return medida_fit
  def retornar_indice(self,elemento, matriz):
    indice = 0
    for i in range(0,len(matriz)):
      if(matriz[i]==elemento):
        return indice
      else:
        indice = indice + 1
  def bernouli_selection(self):
      hijos = []
      i = 0
      while(i != 2):
          hijo = int(random.random() * len(self.poblacion))
          random_range = random.random() * 100
          if(random_range < self.matriz_medidas_fit[hijo]):
              hijos.append(self.poblacion[hijo])
              i = i+1
      return hijos[0], hijos[1]
  def cruzamiento_punto_de_corte(self, padre, madre):
      punto_de_corte = int(random.random() * len(padre))
      adn1 = str(padre[:punto_de_corte])
      adn2 = str(madre[punto_de_corte:])
      return str(adn1+adn2)
  def generate_0_1(self):
      return random.choice('01')
  def mutacion(self, son):
      hijo = son
      val_en_cadena = '1'
      for i in range(1, len(hijo)):
          rand_num = random.random()
          if(rand_num < self.mutation_rate):
              nuevo_alelo = self.generate_0_1()
              val_en_cadena = val_en_cadena + nuevo_alelo
          else:
              val_en_cadena = val_en_cadena + hijo[i]
      return str(val_en_cadena)
  def evolve_main(self):
    for i in range(0,self.total_generations):
      #Asignando todas las medidas fitness
      total_de_medidas_fitness = 0
      for u in range(0,len(self.poblacion)):
        self.matriz_medidas_fit[u] = self.funcion_fitness(self.poblacion[u])
        total_de_medidas_fitness = total_de_medidas_fitness + self.funcion_fitness(self.poblacion[u])
      #Transformar medidas fitness a porcentajes
      for h in range(0,len(self.matriz_medidas_fit)):
          self.matriz_medidas_fit[h] = int(float(self.matriz_medidas_fit[h] * 100)/float(total_de_medidas_fitness))

      #Elitismo (En la posicion cero se guardará el mejor elemento de la generacion pasada)
      indice_elemento_max = self.retornar_indice(max(self.matriz_medidas_fit),self.matriz_medidas_fit)
      elemento_max_fitness = self.poblacion[indice_elemento_max]
      self.poblacion_con_hijos[0] = elemento_max_fitness

      #Main del algoritmo genetico
      for a in range(1,len(self.poblacion)):
        padre, madre = self.bernouli_selection()
        hijo = self.cruzamiento_punto_de_corte(padre, madre)
        hijo = self.mutacion(hijo)
        #El problema de las matrices está aqui
        self.poblacion_con_hijos[a] = hijo
      self.poblacion = self.poblacion_con_hijos



poblacion_definida = ['110110001111','110100001111','110001001111','101100111111']
AG = Algoritmo_Genetico(poblacion_muestra, 'L', 0.2, matriz_hijos)
print "La poblacion inicial: " + str(AG.poblacion)
matriz1_fit = np.zeros(len(AG.poblacion))
matriz2_fit = np.zeros(len(AG.poblacion))
matriz_numeracion = np.zeros(len(AG.poblacion))
for i in range(0,len(AG.poblacion)):
    #print AG.funcion_fitness(AG.poblacion[i])
    matriz1_fit[i] = AG.funcion_fitness(AG.poblacion[i])
    matriz_numeracion[i] = i+1
plt.plot(matriz_numeracion,matriz1_fit,marker ='*', color='red')

AG.evolve_main()

print "La poblacion final: " + str(AG.poblacion)
for i in range(0,len(AG.poblacion)):
    #print AG.funcion_fitness(AG.poblacion[i])
    matriz2_fit[i] = AG.funcion_fitness(AG.poblacion[i])
plt.plot(matriz_numeracion,matriz2_fit,marker ='*', color='green')

plt.show()

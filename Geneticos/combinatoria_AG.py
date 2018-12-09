#-*- coding: utf-8 -*-
import numpy as np
import random
from matplotlib import pyplot as plt
esquema = "1**"
def crear_poblacion(cad):
  combinaciones = []
  counting = cad.count('*')
  for i in range(0,2**counting):
    cadena = cad.replace('*','{}').format(*tuple('{0:b}'.format(i).zfill(counting)))
    combinaciones.append(cadena)
  return combinaciones

def graficar(tamano_x, poblacion, medidas_fit, colour):
    matriz_numeracion = np.zeros(tamano_x)
    for i in range(0,tamano_x):
        matriz_numeracion[i] = i+1
    plt.plot(matriz_numeracion,medidas_fit,marker ='*', color=colour)

#*****************************************************************************

esquema = "1**********1"

poblacion = (crear_poblacion(esquema))
#Definiendo matriz de hijos
matriz_hijos = {"A":["B","C"],"B":["A","E","D"],"C":["A","D","G"],"D":["B","C","I"],"E":["B","F"],"F":["L","K","J","I"],"G":["C","H"],"H":["G","I"],"I":["F","J","D","H"],"J":["F","K","I"],"K":["F","J","L"],"L":["F","K"]}

class Algoritmo_Genetico:
  def __init__(self, poblacion, letra_stop, mutation_rate, matriz_hijos, generaciones):
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
    self.total_generations = generaciones
    #Matriz con los mejores hijos
    self.mejores_hijos_x_generacion = poblacion[:self.total_generations]

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
  def cruzamiento_probabilistics(self, padre_po, madre_po):
      padre = padre_po
      madre = madre_po
      child = '1'
      print padre[1]
      for i in range(1, len(padre)):
          num = random.random()
          if(num < 0.5):
              child = child + padre[i]
          elif(num>=0.5):
              child = child + madre[i]

      return child
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
      for i in range(1, len(hijo)-1):
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
      self.mejores_hijos_x_generacion[i] = elemento_max_fitness
      #Main del algoritmo genetico
      for a in range(1,len(self.poblacion)):
        padre, madre = self.bernouli_selection()
        hijo = self.cruzamiento_punto_de_corte(padre, madre)
        hijo = self.mutacion(hijo)
        #El problema de las matrices está aqui
        self.poblacion_con_hijos[a] = hijo
      self.poblacion = self.poblacion_con_hijos


poblacion_muestra = poblacion[300:400]
AG = Algoritmo_Genetico(poblacion_muestra, 'L', 0.25, matriz_hijos,50)

matriz_fit = np.zeros(len(AG.poblacion))
for i in range(0,len(AG.poblacion)):
    matriz_fit[i] = AG.funcion_fitness(AG.poblacion[i])
graficar(len(AG.poblacion),AG.poblacion,matriz_fit, 'red')
AG.evolve_main()
print "La poblacion final: " + str(AG.poblacion)
for i in range(0,len(AG.poblacion)):
    matriz_fit[i] = AG.funcion_fitness(AG.poblacion[i])
graficar(len(AG.poblacion),AG.poblacion,matriz_fit, 'green')

plt.show()

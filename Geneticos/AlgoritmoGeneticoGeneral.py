import numpy as np
import math
import random
from matplotlib import pyplot as plt

class Genetic_Algorithm:
    def __init__(self, population, ending_goal,mutation_rate, cross_rate=0.3,cruzamiento_method = 'cross_Over'):
        self.cruzamiento_method = cruzamiento_method
        self.mutation_rate = mutation_rate
        self.cross_rate = cross_rate
        self.ending_goal = ending_goal
        self.copia_percentage = []
        self.generations = 0
        self.activation = True
        self.population = population
        self.fitness_elements = np.zeros(len(self.population))


    #Funciones que estan relacionadas con el problema de optimizar funciones
    #Funcion que decodifica las cadenas binario a numero decimal
    def decodificacion_dominio_binario(self, cadena):
        numero = 0
        for i in range(0,len(cadena)):
            numero = numero + (int(cadena[i]) * 2 **(-(i + 1)))
        return numero
    #Funcion de la cual hallaremos su punto maximo, con esta misma funcion calcularemos el fitness de cada elemento
    def calculating_fitness_through_math_function(self, x):
        return ((1 -((float(11)/float(2))* x - float(7)/float(2)) **2)* (math.cos((float(11)/float(2))*x - float(7)/float(2)) + 1)) + 2
    #Funcion de seleccion
    def seleccionando_padres(self):
        self.suma_calificaciones_poblacion = 0
        #Primero calcularemos la sumatoria de las medidas fitness de cada elemento de la poblacion
        for i in range(0,len(self.population)):
            numero_decodificado = self.decodificacion_dominio_binario(self.population[i])
            self.suma_calificaciones_poblacion = self.suma_calificaciones_poblacion + self.calculating_fitness_through_math_function(numero_decodificado)
        indice = 0
        padres = []
        #Proceso de seleccion de hijos
        while(indice != 2):
            primer_hijo = self.population[int(random.random() * len(self.population))]
            num_aleatorio = random.random()
            c = num_aleatorio * self.suma_calificaciones_poblacion
            calificacion_acumulada = 0
            calificacion_acumulada = calificacion_acumulada + self.calculating_fitness_through_math_function(self.decodificacion_dominio_binario(primer_hijo))
            if(calificacion_acumulada > c):
                padres.append(primer_hijo)
                indice = indice + 1
        return padres[0], padres[1]


    #Calculando medida fintess para string problema
    def calculating_fitness_letras(self,element):
        fitness = 0
        for i in range(0,len(element)):
            if(element[i] == self.ending_goal[i]):
                fitness = fitness + 1
        return fitness
    #Calculando medida fitness para todos los elementos mas su porcentaje
    def asignar_medida_fit_percentage_relacionado_con_bernouli(self):
        percentage_calc = 0
        for i in range(0,len(self.population)):
            self.fitness_elements[i] = self.calculating_fitness_letras(self.population[i])
            percentage_calc = percentage_calc + self.calculating_fitness_letras(self.population[i])
        self.copia_percentage = self.fitness_elements
        for i in range(0,len(self.population)):
            self.fitness_elements[i] = int(float(self.fitness_elements[i] * 100) / float(percentage_calc))

    #Metodos para seleccionar padre y madre
    def bernouli_selection(self):
        hijos = []
        i = 0
        while(i != 2):
            hijo = int(random.random() * len(self.population))
            random_range = random.random() * 100
            if(random_range < self.fitness_elements[hijo]):
                hijos.append(self.population[hijo])
                i = i+1
        return hijos[0], hijos[1]

    #Metodos de cruzamiento para los progenitores
    def cross_Over(self,father, mother):
        mid_point = int(float(len(father)) / float(2))
        adn1 = father[:mid_point]
        adn2 = mother[mid_point:]
        return adn1+adn2
    def cross_over_probabilistic(self, father, mother):
        num = int(random.random() * (len(father) - 1))
        adn1 = father[:num]
        adn2 = mother[num:]
        return adn1 + adn2
    def cross_over_jumped(self, father, mother):
        child = self.population[5]
        for i in range(0, len(father)):
            if(i % 2 == 0):
                child.replace(child[i], father[i])
            else:
                child.replace(child[i], mother[i])
        return child
    def cross_over_volado(self,father, mother):
    	i = 0
    	child = self.population[5]
    	while(i<len(father)):
    		random_numer = int(random.random()*2)
    		if(random_numer == 1):
    			child.replace(child[i], father[i])
    			i = i + 1
    		elif(random_numer == 2):
    			child.replace(child[i], mother[i])
    			i = i + 1
    	return child
    def cruzamiento(self, vater, mutter):
        if(self.cruzamiento_method == 'cross_Over'):
            sohn = self.cross_Over(vater, mutter)
        elif(self.cruzamiento_method == 'cross_over_probabilistic'):
            sohn = self.cross_over_probabilistic(vater, mutter)
        elif(self.cruzamiento_method == 'cross_over_jumped'):
            sohn = self.cross_over_jumped(vater, mutter)
        elif(self.cruzamiento_method == 'cross_over_volado'):
            sohn = self.cross_over_volado(vater, mutter)
        return sohn

    #Metodos de mutacion
    def mutate_letra(self,son):
        hijo = son
        for i in range(0, len(son)):
            if(random.random() < self.mutation_rate):
                hijo = hijo.replace(hijo[i], random.choice('abcdefghijklmnopqrstuvwxyz '))
        return hijo
    def mutate_bin_number(self,son):
        hijo = son
        for i in range(0, len(son)):
            if(random.random() < self.mutation_rate):
                if(hijo[i] == '0'):
                    hijo = hijo.replace(hijo[i], '1')
                else:
                    hijo = hijo.replace(hijo[i], '0')
        return hijo

    #Evolucion principal
    def evolve(self):
        while(self.activation==True and self.generations < 20):
            #self.asignar_medida_fit_percentage_relacionado_con_bernouli() #Para problema de maximizar funcion
            for i in range(0,len(self.population)):
                #vater, mutter = self.bernouli_selection()       #Para problema de cadenas
                vater, mutter = self.seleccionando_padres()      #Para problema de maximizar funcion
                sohn = self.cruzamiento(vater, mutter)
                #sohn = self.mutate_letra(sohn)                 #Para problema de cadenas
                sohn = self.mutate_bin_number(sohn)
                self.population[i] = sohn                       #Para problema de maximizar funcion
                '''
                if(sohn == self.ending_goal):
                    self.activation = False
                    print self.generations
                '''
            self.generations = self.generations + 1

'''
    Codificando el dominio
    numero   E.Dominio  f(x)

    000        0.000    1.285
    001        0.125    1.629
    010        0.250    0.334
    011        0.375    0.791
    100        0.500    2.757
    101        0.625    3.990
    110        0.750    3.103
    111        0.875    1.092
'''
'''
#Funcion para crear numero binario en cada elemento de la poblacion
def numero_binario():
    numero = random.random()
    if(numero < 0.5):
        return 0
    else:
        return 1
#Funcion para crear la pobracion
def crear_poblacion(numero_de_poblacion_inicial, len_elemento):
    poblacion = []
    for i in range(0,numero_de_poblacion_inicial):
        numero = str(numero_binario())
        for i in range(0, len_elemento-1):
            numero = numero + str(numero_binario())
        poblacion.append(numero)
    return poblacion

#Vamos a comenzar con el proceso del algoritmo genetico

poblacion = crear_poblacion(5, 3)
AG = Genetic_Algorithm(poblacion, '101', 0.01, 0.3,'cross_over_probabilistic')


print AG.population
AG.evolve()
print AG.population
'''

'''
    #EJEMPLO CON LAS CADENAS

    population = []
    def new_Letter():
        return random.choice('abcdefghijklmnopqrstuvwxyz ')
    for i in range(0,100):
        new_element = new_Letter()
        for j in range(0,4):
            new_element = new_element + new_Letter()
        population.append(new_element)

    AG = Genetic_Algorithm(population, 'diego', 0.01, 0.3, 'cross_over_probabilistic')
    print AG.population
    AG.evolve()
    print AG.population
'''

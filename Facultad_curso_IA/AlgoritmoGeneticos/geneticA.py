import numpy as np
from matplotlib import pyplot as plt
import random

class Genetic_Algorith:
    def __init__(self, mutation_rate, population, ending_goal, cruzamiento_method = 'cross_Over'):
        self.cruzamiento_method = cruzamiento_method
        self.mutation_rate = mutation_rate
        self.number_population = population
        self.ending_goal = ending_goal
        self.population = []
        self.copia_percentage = []
        self.generations = 0
        self.activation = True
        self.population_pool = []
        for i in range(0,self.number_population):
            new_element = self.new_Letter()
            for j in range(0,len(self.ending_goal)-1):
                new_element = new_element + self.new_Letter()
            self.population.append(new_element)
        self.fitness_elements = np.zeros(len(self.population))
    #Metodo inicializador de variables principales del algoritmo
    def new_Letter(self):
        #return random.choice('1234567890')
        return random.choice('abcdefghijklmnopqrstuvwxyz ')
    #Calculando medida fintess para cada elemento
    def calculating_fitness(self,element):
        fitness = 0
        for i in range(0,len(element)):
            if(element[i] == self.ending_goal[i]):
                fitness = fitness + 1
        return fitness
    #Calculando medida fitness para todos los elementos mas su porcentaje
    def asignar_medida_fit_percentage(self):
        self.population_pool = []
        percentage_calc = 0
        for i in range(0,len(self.population)):
            self.fitness_elements[i] = self.calculating_fitness(self.population[i])
            percentage_calc = percentage_calc + self.calculating_fitness(self.population[i])
        self.copia_percentage = self.fitness_elements
        for i in range(0,len(self.population)):
            self.fitness_elements[i] = int(float(self.fitness_elements[i] * 100) / float(percentage_calc))
    #Metodos para seleccionar padre y madre
    def filling_pool(self):
        for i in range(0,len(self.population)):
            for j in range(0,int(self.fitness_elements[i])):
                self.population_pool.append(self.population[i])
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
    #Metodos de mutacion
    def mutate(self,son):
        hijo = son
        for i in range(0, len(son)):
            if(random.random() < self.mutation_rate):
                hijo = hijo.replace(hijo[i], self.new_Letter())
        return hijo
    #Evolucion principal
    def evolve(self):
        while(self.activation==True):
            self.asignar_medida_fit_percentage()
            #self.filling_pool() #En caso de usar el metodo de la alberca
            for i in range(0,len(self.population)):
                #vater = self.population_pool[int(random.random()*len(self.population_pool))] #En caso de usar el metodo de alberca
                #mutter = self.population_pool[int(random.random()*len(self.population_pool))] #En caso de usar el metodo de alberca
                vater, mutter = self.bernouli_selection()
                if(self.cruzamiento_method == 'cross_Over'):
                    sohn = self.cross_Over(vater, mutter)
                elif(self.cruzamiento_method == 'cross_over_probabilistic'):
                    sohn = self.cross_over_probabilistic(vater, mutter)
                elif(self.cruzamiento_method == 'cross_over_jumped'):
                    sohn = self.cross_over_jumped(vater, mutter)
                sohn = self.mutate(sohn)
                if(sohn == self.ending_goal):
                    self.activation = False
                    print self.generations
                    #self.graficar()
                    print sohn
                self.population[i] = sohn
                #print sohn + '      Numero de generacion:   '+ str(self.generations)
            self.generations = self.generations + 1
    #Metodo para graficar los elementos y su progreso
    def graficar(self):
        if((self.generations % 10 == 0) or (self.generations == 1) or (self.activation == False)):
            for i in range(0, len(self.population)):
                plt.plot(i, self.copia_percentage[i], marker = 'o', color = 'red')
                plt.plot(i, len(self.ending_goal))
            plt.show()


#Aplicando la clase del algoritmo para evolucionar una palabra
AG = Genetic_Algorith(0.05,100,'diego', 'cross_over_probabilistic')
print AG.population
print AG.cruzamiento_method
AG.evolve()
print AG.population

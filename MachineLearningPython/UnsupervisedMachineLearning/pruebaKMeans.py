import numpy as np
from matplotlib import pyplot as plt
import math
import random
import pandas as pd

data = np.random.random_sample((10000, 2))*10
clusters = 3
pruebasSize = 20
prueba = np.array([5,1])
'''
df = pd.read_csv('breast-cancer.csv')
df.replace('?',-99999, inplace = True)
df.drop(['id'], 1, inplace = True)
X = np.array(df.drop(['class'], 1))
'''
class KMeans:
    def __init__(self,data,clusters,pruebasSize):
        self.data = data
        self.clusters = clusters
        self.pruebasSize = pruebasSize
        self.centroide = []
        for i in range(0,clusters):
            self.centroide.append(data[i])
    def eucledean_distance(self,A,B):
      valores = []
      for i in range(0,len(A)):
        valores.append((A[i] - B[i])**2)
      suma = 0
      for i in range(0,len(valores)):
        suma = suma + valores[i]
      distancia = math.sqrt(suma)
      return distancia
    def fit(self):
        for i in range(0,self.pruebasSize):
            self.diccionario = {}
            for i in range(0,self.clusters):
                self.diccionario[i] = []

            for i in range(len(self.data)):
                eucledean_distances = []
                for u in range(len(self.centroide)):
                    distancia = self.eucledean_distance(data[i],self.centroide[u])
                    #distancia = math.sqrt((self.data[i][0] - self.centroide[u][0])**2 + (self.data[i][1]-self.centroide[u][1])**2)
                    eucledean_distances.append(distancia)
                self.diccionario[eucledean_distances.index(min(eucledean_distances))].append(self.data[i])

            for i in range(0,len(self.diccionario)):
                promedioX = 0
                promedioY = 0
                for u in range(0, len(self.diccionario[i])):
                    promedioX = promedioX + self.diccionario[i][u][0]
                    promedioY = promedioY + self.diccionario[i][u][1]
                self.centroide[i][0] = float(promedioX) / float(len(self.diccionario[i]))
                self.centroide[i][1] = float(promedioY) / float(len(self.diccionario[i]))
    def predecir(self,prueba):
        eucledean_distances = []
        for u in range(len(self.centroide)):
            distancia = math.sqrt((prueba[0] - self.centroide[u][0])**2 + (prueba[1]-self.centroide[u][1])**2)
            eucledean_distances.append(distancia)
        self.diccionario[eucledean_distances.index(min(eucledean_distances))].append(prueba)
        print eucledean_distances.index(min(eucledean_distances))
    def graficar(self):
        MatrizClusters = []
        colors = ['green','red','blue','purple','yellow']
        for i in range(0,len(self.diccionario)):
            MatrizClusters.append(np.array(self.diccionario[i]))
        self.centroide = np.array(self.centroide)
        for i in range(len(MatrizClusters)):
            plt.scatter(MatrizClusters[i][:,0],MatrizClusters[i][:,1], s = 150, color = colors[i])
        plt.scatter(self.centroide[:,0], self.centroide[:,1], color = 'black', marker = '*')
        plt.show()

clf = KMeans(data,clusters,pruebasSize)
clf.fit()
clf.predecir(prueba)
clf.graficar()

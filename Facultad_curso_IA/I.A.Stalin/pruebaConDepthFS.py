import random

padre_hijo = [(0,1),(0,4),(1,0),(1,2),(1,5),(1,7),(2,1),(3,4),(3,6),(4,0),(4,3),(4,11),(5,1),(5,9),(6,3),(6,9),
            (6,10),(7,1),(7,8),(7,11),(8,7),(8,11),(9,5),(9,6),(9,13),(10,6),(10,14),(10,17),(11,7),(11,8),(11,12),
            (11,15),(12,11),(12,16),(12,23),(13,9),(13,17),(14,10),(14,15),(15,11),(15,14),(16,12),(16,20),(17,10),
            (17,13),(17,18),(18,17),(18,21),(19,20),(20,16),(20,19),(21,18),(21,22),(21,25),(22,21),(22,23),(23,12),
            (23,22),(23,24),(23,26),(23,27),(24,23),(24,28),(25,21),(25,26),(26,23),(26,25),(26,27),(27,23),(27,26),(27,28),(28,24),(28,27)]


adjacentList = [[] for vertex in range(29)] #7 es el numero de vertices

for i in padre_hijo:
  adjacentList[i[0]].append(i[1])

def desorganizar(array):
    array = random.shuffle(array)
    return array

#Funcion que tiene el algoritmo de Depth-First Search

def camino(estadoInicial, estadoFinal, adjacentList):
    estado = True
    pilaHijos = [estadoInicial]
    listaCamino = []
    while(estado):
      numNodo = pilaHijos.pop()
      if(len(listaCamino)!= 0 and not numNodo in adjacentList[listaCamino[len(listaCamino)-1]]):
          estado = False
      #Aqui tengo la extraccion del ultimo nodo de la pila y tengo el ultimo nodo, tengo que combrobar si ambos son hermanos
      if(not numNodo in listaCamino):
          if(numNodo == estadoFinal):
            estado = False
          for neighbor in adjacentList[numNodo]:
            pilaHijos.append(neighbor)
          listaCamino.append(numNodo)
    return listaCamino

#Definicion de arrays que tendran los caminos y sus longitudes.
caminos = []
indexCaminos = []
coeficiente = 1000
i = 0

#Ubicandondo puntos de inicio y final
#punto_inicio = 25
#punto_final = 19

punto_inicio = int(raw_input("Ingresa el punto de inicio: "))
punto_final  = int(raw_input("Ingresa el punto final: "))

while i < coeficiente:
    for subList in range(len(adjacentList)):
        desorganizar(adjacentList[subList])
    camino1 = camino(punto_inicio,punto_final,adjacentList)
    if(camino1[len(camino1)-1] == punto_final and not camino1 in caminos):
        caminos.append(camino1)
        indexCaminos.append(len(camino1))
        print camino1
    i = i+1

print("El camino mas optimo es: ",caminos[indexCaminos.index(min(indexCaminos))])

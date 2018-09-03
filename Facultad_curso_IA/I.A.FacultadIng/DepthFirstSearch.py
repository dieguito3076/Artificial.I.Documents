edgeList = [(0,1),(0,2),(1,0),(1,3),(2,0),(2,4),(2,5),(3,1),(4,2),(4,6),(5,2),(6,4)]
adjacentList = [[] for vertex in range(7)] #7 es el numero de vertices

for edge in edgeList:
  adjacentList[edge[0]].append(edge[1])

visitedList = []
stack = [0]
while stack:
  current = stack.pop()
  for neighbor in adjacentList[current]:
    if not neighbor in visitedList:
      stack.append(neighbor)
    visitedList.append(current)

print (visitedList)

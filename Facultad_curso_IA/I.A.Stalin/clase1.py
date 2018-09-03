#Ejercicio ALGORITMO DFS
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from functools import reduce
import random

seq = list(range(0,16))

actions = ['E','W','N','S']

x_mask = lambda i: 15<<(4*i)

extract = lambda i,c: (c&(x_mask(i)))>>(4*i)

e_most = lambda z: (z%4)==3

w_most = lambda z: (z%4)==0

n_most = lambda z: z<=3

s_most = lambda z:z>=12

valid_moves = {i:list(filter(lambda action:\
((not action=='E') or (not e_most(i))) and \
((not action=='W') or (not w_most(i))) and \
((not action=='S') or (not s_most(i))) and \
((not action=='N') or (not n_most(i))),actions)) for i in seq}

def move_east(puzzle):
    if(not e_most(puzzle.zero)):
        puzzle.zero += 1;
        mask = x_mask(puzzle.zero)
        puzzle.configuration = \
        (puzzle.configuration&mask)>>4 | \
        (puzzle.configuration&~mask)

def move_west(puzzle):
    if(not w_most(puzzle.zero)):
        puzzle.zero -= 1;
        mask = x_mask(puzzle.zero)
        puzzle.configuration = \
        (puzzle.configuration&mask)<<4 | \
        (puzzle.configuration&~mask)

def move_north(puzzle):
    if(not n_most(puzzle.zero)):
        puzzle.zero -= 4;
        mask = x_mask(puzzle.zero)
        puzzle.configuration = \
        (puzzle.configuration&mask)<<16 | \
        (puzzle.configuration&~mask)

def move_south(self):
    if(not s_most(self.zero)):
        self.zero += 4;
        mask = x_mask(self.zero)
        self.configuration = \
        (self.configuration&mask)>>16 | \
        (self.configuration&~mask)

class Puzzle:

    def __init__(self, parent=None, action =None, depth=0):
        self.parent = parent
        self.depth = depth
        if(parent == None):
            self.configuration =  \
                reduce(lambda x,y: x | (y << 4*(y-1)), seq)
            self.zero = 15
        else:
            self.configuration = parent.configuration
            self.zero = parent.zero
            if(action != None):
                self.move(action)

    def __str__(self):
        return '\n'+''.join(list(map(lambda i:\
        format(extract(i,self.configuration)," x")+\
        ('\n' if (i+1)%4==0 else ''),seq)))+'\n'

    def __repr__(self):
        return self.__str__()

    def __eq__(self,other):
        return (isinstance(other, self.__class__)) and \
        (self.configuration==other.configuration)

    def __ne__(self,other):
        return not self.__eq__(other)

    def __lt__(self,other):
        return self.depth < other.depth

    def __hash__(self):
        return hash(self.configuration)

    def move(self,action):
        if(action =='E'):
            move_east(self)
        if(action =='W'):
            move_west(self)
        if(action =='N'):
            move_north(self)
        if(action =='S'):
            move_south(self)
        return self


    @staticmethod
    def to_list(puzzle):
        return [extract(i,puzzle.configuration) for i in seq]

    def shuffle(self,n):
        for i in range(0,n):
            self.move(random.choice(valid_moves[self.zero]))
        return self

    def expand(self):
        #filtering the path back to parent
        return list(filter(lambda x: \
        (x!=self.parent), \
        [Puzzle(self,action,self.depth+1) \
        for action in valid_moves[self.zero]]))

p = Puzzle()
p.shuffle(20)

class DFS:
    @staticmethod
    #El stop es una función
    def solve(start,stop):
        if(stop(start)):
            return ruta(start)
        agenda = [start]
        x = set() #Lista de expandidos x
        while agenda:
            n = agenda.pop()
            x.add(n)
            for h in n.expand():
                if(stop(h)):
                    return ruta(h)
                if( h not in x):
                    agenda.append(h)
DFS1 = DFS()
DFS1.solve(p,lambda n: n==Puzzle())


class BFS:
    @staticmethod
    #El stop es una función
    def solve(start,stop):
        if(stop(start)):
            return ruta(start)
        agenda = deque()
        agenda.append(start)
        x = set() #Lista de expandidos x
        while agenda:
            n = agenda.popleft()
            x.add(n)
            for h in n.expand():
                if(stop(h)):
                    return ruta(h)
                if( h not in x):
                    agenda.append(h)

for i in range(0,10): agenda.append(i)
BFS1 = BFS()
BFS1.solve(p,lambda n: n==Puzzle())

class BidiSearch:
    @staticmethod
    def solve(start, end):
        x = set()
        f_front = {start:start}
        b_front = {end:end}
        b_t_front = {}
        f_t_front = {}
        while f_front or b_front:
            #Expansion hacia adelante
            for i in f_front:
                if n not in x:
                    if n in b_front :
                        return ruta
                    f_t_front[n] = n
            f_front = f_t_front
            #Expansion hacia atras
            for i in b_front:
                if n not in x:
                    if n in f_front :
                        return ruta
                    b_t_front[n] = n
            b_front = b_t_front

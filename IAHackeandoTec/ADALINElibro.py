#-*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import math
import scipy as sc #funciones cientificas

class AdalineGD(object):
    def __init__(self, eta = 0.1, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, Y):
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0])
    def activation(self, X):
        return self.net_input(X)
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
adal = AdalineGD(n_iter = 10, eta = 0.01,).fit(X, y)
ax[0].plot(range(1, len(adal.cost_) + 1), np.log10(adal.cost_), marker = 'o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')
ada2 = AdalineGD(n_iter = 10, eta = 0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker = 'o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum- squared error')
ax[1].set_title('Adaline learning rate 0.0001')
plt.show()

'''
array_puntos = np.array([[0,0],[1,0],[0,1],[1,1]])
for i in range(4):
    plt.plot(array_puntos[i][0], array_puntos[i][1], "o", c = "red")

plt.show()
'''

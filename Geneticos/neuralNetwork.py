'''

        1. # Input Layer
        2. # Hidden Layer
        3. # Output Layer

    import numpy as np
    from numpy import exp
    from matplotlib import pyplot as plt

    np.random.seed(100)

    class NeuralNetwork:

        def __init__(self, number_inputs, number_hidden, number_output):
            self.number_hidden = number_hidden
            self.number_output = number_output
            self.number_inputs = number_inputs
            self.learning_rate = 0.001
            self.errores = []
        def sigmoid(self, x):
        	return 1/(1+ exp(-x))
        def feedForward(self, X):
            #Codigo que considera el bias
                #X = np.append(X, 1)
                #self.w1 = 2.0 * np.random.random((self.number_hidden,self.number_inputs+1))
                #self.w2 = 2.0 * np.random.random((self.number_output,self.number_hidden+1))
                #weight_multiplication_inputs = np.append(self.sigmoid(np.dot(self.w1,X)),1)
            self.w1 = 2.0 * np.random.random((self.number_hidden,self.number_inputs))
            self.w2 = 2.0 * np.random.random((self.number_output,self.number_hidden))
            self.weight_multiplication_inputs = self.sigmoid(np.dot(self.w1,X))
            self.output = self.sigmoid(np.dot(self.w2,self.weight_multiplication_inputs))
            return self.output

        def train(self, inputs, labels):
            outputs = self .feedForward(inputs)
            output_errors = labels - outputs
            hidden_weights_transposed = self.w2.T
            hidden_errors = np.dot(hidden_weights_transposed, output_errors)

            #Computing the gradients
            derivate_Loss_h2 = -(labels-outputs)
        	derivate_h2_z2 = np.dot((1-outputs),outputs)
        	h1_Transpose = self.weight_multiplication_inputs.T
        	derivate_Loss_W2 = derivate_Loss_h2 * derivate_h2_z2 * h1_Transpose
        	self.w2 = self.w2 - (self.learning_rate * derivate_Loss_W2)

        	d1_1_1 = np.dot(np.dot((-(labels-outputs)),self.w2),np.dot(outputs,(1-outputs)))
        	d2_2_1 =  (1-self.weight_multiplication_inputs) * self.weight_multiplication_inputs
        	x_Tranpose = [[inputs[0]],[inputs[1]]]
        	derivate_Loss_W1 = d1_1_1 * d2_2_1 * x_Tranpose
        	self.w1 = self.w1 - (self.learning_rate * derivate_Loss_W1)


        	z2 = np.dot(self.w2,self.weight_multiplication_inputs)
        	inputs = np.array(inputs)
        	m = inputs.shape[0]
        	Delta_2 = (labels - z2)
        	error = np.sum(Delta_2.T ** 2) / (2 * m)
        	self.errores.append(error)

    neuralN = NeuralNetwork(2,2,1)
    training_data = np.array([[0,1],[1,0],[0,0],[1,1]])
    y = np.array([1,1,0,0])
    neuralN.train([0,0], 0)

    for i in range(0,100):
    	for i in range(0,4):
    		neuralN.train(training_data[i], y[i])


    print neuralN.feedForward([1,1])
    print neuralN.feedForward([0,0])
    print neuralN.feedForward([1,0])
    print neuralN.feedForward([0,1])


    for i in range(0, len(neuralN.errores)):
    	plt.plot(i,neuralN.errores[i], marker = '*', color = 'red')
    plt.show()
#-----------------------------------------------------------------------------
'''

import numpy as np
from numpy import exp
from matplotlib import pyplot as plt
from AlgoritmoGeneticoGeneral import Genetic_Algorithm
#np.random.seed(seed=100)

weights_UNO = []
weights_DOS = []

def sigmoid(x):
  return 1/(1+ exp(-x))

learning_rate = 0.001

errores_red = []

number_inputs = 2
number_hidden = 3
number_output = 1

X = np.array([1, 1])
y = np.array([0])

for i in range(0,10):

    #Inicializacion de pesos sinapticos

    w1 =  np.random.random((number_inputs,number_hidden))
    w2 =  np.random.random((number_hidden, number_output))

    for i in range(0,3000):

     #feedForward

      z1 = np.dot(X, w1)
      h1 = sigmoid(z1)
      z2 = np.dot(h1, w2)
      h2 = sigmoid(z2)

      #Calculating error

      Delta_2 = (y - z2)
      m = X.shape[0]
      error = np.sum(Delta_2.T ** 2) / (2 * m)
      errores_red.append(error)

      #BackPropagation

      dLoss_dh2 = -(y - h2)
      resta_uno_h2 = 1 - h2
      dh2_dz2 = resta_uno_h2 * h2
      matriz_trans_h1 = [[h1[0]], [h1[1]], [h1[2]]]
      dLoss_dw2 = (dLoss_dh2 * dh2_dz2) * matriz_trans_h1
      w2 = w2 - learning_rate * dLoss_dw2

      dLoss_dh1 = np.dot(w2, dLoss_dh2 * dh2_dz2)
      dh1_dz1 = h1* (1 - h1)
      x_trans = [[X[0]], [X[1]]]
      dLoss_dw1 = dLoss_dh1 * dh1_dz1 * x_trans
      w1 = w1 - learning_rate * dLoss_dw1

    weights_UNO.append(w1)
    weights_DOS.append(w2)

poblacion_GENERAL = []
for i in range(0, len(weights_UNO)):
    matriz = [weights_UNO[i], weights_DOS[i]]
    poblacion_GENERAL.append(matriz)

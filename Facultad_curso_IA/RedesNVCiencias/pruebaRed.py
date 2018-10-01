import numpy as np
from numpy import exp
from matplotlib import pyplot as plt
import random

np.random.seed(100)

class Neural_Network:
    def __init__(self,input_dim, hidden_dim, output_dim, n_iter = 1000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = 0.001
        self.n_iter = n_iter
        self.errores = []
        self.indexes = []
    def sigmoid(self,x):
        return 1/(1+ exp(-x))
    def train(self,X, y):
        self.w1 = 2.0 * np.random.random((self.input_dim, self.hidden_dim))
        self.w2 = 2.0 * np.random.random((self.hidden_dim,self.output_dim))
        for i in range(1,self.n_iter):

            h1 = self.sigmoid(np.dot(X,self.w1))
            h2 = self.sigmoid(np.dot(h1,self.w2))

            derivate_Loss_h2 = -(y-h2)
            derivate_h2_z2 = np.dot(h2,(1-h2))
            h1_Transpose = [[h1[0]],[h1[1]],[h1[2]]]
            derivate_Loss_W2 = derivate_Loss_h2 * derivate_h2_z2 * h1_Transpose
            self.w2 = self.w2 - (self.learning_rate * derivate_Loss_W2)


            d1_1_1 = np.dot(np.dot(self.w2,(-(y-h2))),np.dot((1-h2),h2))
            d2_2_1 =  (1-h1) * h1
            x_Tranpose = [[X[0]],[X[1]]]
            #x_Tranpose = X.T
            derivate_Loss_W1 = d1_1_1 * d2_2_1 * x_Tranpose
            self.w1 = self.w1 - (self.learning_rate * derivate_Loss_W1)



            m = X.shape[0]
            Delta_2 = (y.T - np.dot(h1,self.w2))
            error = np.sum(Delta_2.T ** 2) / (2 * m)
            self.errores.append(error)
            self.indexes.append(i)

    def guess(self,X):
        z1 = np.dot(X,self.w1)
        h1 = self.sigmoid(z1)
        z2 = np.dot(h1,self.w2)
        h2 = self.sigmoid(z2)
        return h2


y = np.array([1,0])
X = np.array([1.7640,0.0015])

x_train = np.array([[1,1],[0,0],[1,0],[0,1]])
y_labels = np.array([0,0,1,1])

clf = Neural_Network(2,3,1)

for i in range(0,4):
    clf.train(x_train[i], y_labels[i])

print clf.guess([0,0])
print clf.guess([1,0])
print clf.guess([1,1])
print clf.guess([0,1])


print clf.w1
print clf.w2
'''
clf.train(X, y)
for i in range(0,len(clf.errores)):
    plt.plot(clf.indexes[i],clf.errores[i], marker = 'o', color = 'red')
plt.xlabel('iterations')
plt.ylabel('error')
plt.show()


import numpy as np
import ipdb
from scratch_mlp import utils
utils.reset_folders()

def load_XOR_data(N=300):
    rng = np.random.RandomState(0)
    X = rng.randn(N, 2)
    y = np.array(np.logical_xor(X[:, 0] > 0, X[:, 1] > 0), dtype=int)
    y = np.expand_dims(y, 1)
    y_hot_encoded = []

    for x in y:
        if x == 0:
            y_hot_encoded.append([1,0])
        else:
            y_hot_encoded.append([0, 1])
    return X, np.array(y_hot_encoded)

def sigmoid(z, first_derivative=False):
    if first_derivative:
        return z*(1.0-z)
    return 1.0/(1.0+np.exp(-z))

def tanh(z, first_derivative=True):
    if first_derivative:
        return (1.0-z*z)
    return (1.0-np.exp(-z))/(1.0+np.exp(-z))

def inference(data, weights):
    h1 = sigmoid(np.matmul(data, weights[0]))
    logits = np.matmul(h1, weights[1])
    probs = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def run():
    #size of minibatch: int(X.shape[0])
    N = 50
    X, y = load_XOR_data(N=300)
    input_dim = int(X.shape[1])
    hidden_dim = 10
    output_dim = 2
    num_epochs = 1000000
    learning_rate= 1e-3
    reg_coeff = 1e-6
    losses = []
    accuracies=[]

    #---------------------------------------------------------------------------------------------------------------
    # Initialize weights:
    np.random.seed(2017)
    w1 = 2.0*np.random.random((input_dim, hidden_dim))-1.0      #w0=(2,hidden_dim)
    w2 = 2.0*np.random.random((hidden_dim, output_dim))-1.0     #w1=(hidden_dim,2)

    #Calibratring variances with 1/sqrt(fan_in)
    w1 /= np.sqrt(input_dim)
    w2 /= np.sqrt(hidden_dim)
    for i in range(num_epochs):

        index = np.arange(X.shape[0])[:N]
        #is want to shuffle indices: np.random.shuffle(index)

        #---------------------------------------------------------------------------------------------------------------
        # Forward step:
        h1 = sigmoid(np.matmul(X[index], w1))                   #(N, 3)
        logits = sigmoid(np.matmul(h1, w2))                     #(N, 2)
        probs = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
        h2 = logits

        #---------------------------------------------------------------------------------------------------------------
        # Definition of Loss function: mean squared error plus Ridge regularization
        L = np.square(y[index]-h2).sum()/(2*N) + reg_coeff*(np.square(w1).sum()+np.square(w2).sum())/(2*N)

        losses.append([i,L])

        #---------------------------------------------------------------------------------------------------------------
        # Backward step: Error = W_l e_l+1 f'_l
        #       dL/dw2 = dL/dh2 * dh2/dz2 * dz2/dw2
        dL_dh2 = -(y[index] - h2)                               #(N, 2)
        dh2_dz2 = sigmoid(h2, first_derivative=True)            #(N, 2)
        dz2_dw2 = h1                                            #(N, hidden_dim)
        #Gradient for weight2:   (hidden_dim,N)x(N,2)*(N,2)
        dL_dw2 = dz2_dw2.T.dot(dL_dh2*dh2_dz2) + reg_coeff*np.square(w2).sum()

        #dL/dw1 = dL/dh1 * dh1/dz1 * dz1/dw1
        #       dL/dh1 = dL/dz2 * dz2/dh1
        #       dL/dz2 = dL/dh2 * dh2/dz2
        dL_dz2 = dL_dh2 * dh2_dz2                               #(N, 2)
        dz2_dh1 = w2                                            #z2 = h1*w2
        dL_dh1 =  dL_dz2.dot(dz2_dh1.T)                         #(N,2)x(2, hidden_dim)=(N, hidden_dim)
        dh1_dz1 = sigmoid(h1, first_derivative=True)            #(N,hidden_dim)
        dz1_dw1 = X[index]                                      #(N,2)
        #Gradient for weight1:  (2,N)x((N,hidden_dim)*(N,hidden_dim))
        dL_dw1 = dz1_dw1.T.dot(dL_dh1*dh1_dz1) + reg_coeff*np.square(w1).sum()

        #weight updates:
        w2 += -learning_rate*dL_dw2
        w1 += -learning_rate*dL_dw1
        if True: #(i+1)%1000==0:
            y_pred = inference(X, [w1, w2])
            y_actual = np.argmax(y, axis=1)
            accuracy = np.sum(np.equal(y_pred,y_actual))/len(y_actual)
            accuracies.append([i, accuracy])

        if (i+1)% 10000 == 0:
            print('Epoch %d\tLoss: %f Average L1 error: %f Accuracy: %f' %(i, L, np.mean(np.abs(dL_dh2)), accuracy))
            save_filepath = './scratch_mlp/plots/boundary/image_%d.png'%i
            text = 'Batch #: %d    Accuracy: %.2f    Loss value: %.2f'%(i, accuracy, L)
            utils.plot_decision_boundary(X, y_actual, lambda x: inference(x, [w1, w2]),
                                         save_filepath=save_filepath, text = text)
            save_filepath = './scratch_mlp/plots/loss/image_%d.png' % i
            utils.plot_function(losses, save_filepath=save_filepath, ylabel='Loss', title='Loss estimation')
            save_filepath = './scratch_mlp/plots/accuracy/image_%d.png' % i
            utils.plot_function(accuracies, save_filepath=save_filepath, ylabel='Accuracy', title='Accuracy estimation')

run()
'''

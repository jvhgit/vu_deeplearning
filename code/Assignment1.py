"""
Name: Jens van Holland
Course: Deep Learning
Assigment: 1
Date: 11-11-2020

The layout of the scipt:

- The first few functions are for training a network based on scalar calculations.
- This network is used to answer question 2 and 3 (Q1 & Q2).

Next,
- I made a network based on classes: Layer, Activation, Optimizer and Loss,
  this makes life easier.
- With these classes I implemented the network of Q4.
- Q5 is answered with some analysis
- A second analysis is done in the same way

"""

"""
First import some libraries, not that numpy is not yet imported.
"""
import plotly.graph_objects as go
import math
from random import gauss as g
from plotly.subplots import make_subplots
from random import seed

"""
Initializing the weights with two options: random or given.
"""
def initialize_weights(random = False):
    """
    Initializes the weights according to the assigment for Q2.
    """
    if not random :
        W,V = [[1.,1.,1.],[-1.,-1.,-1.]] , [[1.,1.],[-1.,-1.],[-1.,-1.]]
    elif random:
        W = [[g(0,1),g(0,1),g(0,1)],[g(0,1),g(0,1),g(0,1)]]
        V = [[g(0,1),g(0,1)],[g(0,1),g(0,1)],[g(0,1),g(0,1)]]
    b,c = [0.,0.,0.], [0.,0.,0.]
    return(W,V,b,c)

"""
Sigmoid function for scalars.
"""
def sigmoid(a): return(1/(1+math.exp(-a)))

"""
Loss function according to the slides of lecture 2.
"""
def loss_forward(outputs,y): return(-1*math.log(outputs[y]))

def loss_backward(outputs, y): 
    """
    Loss derivative function, according to the report. (-log(y_c), c:= true class of instance)
    """
    for i in range(2):
        if i == y:
            outputs[i] = -1/outputs[i]
        else: outputs[i] = 0
    return(outputs)

def forward_backward(X,y,W,V,b,c,print_params = False, lr = 0.00001, compute_loss = False, update = False):
    """
    This function is the main function. It does the forward and back propagation.
    This function is called two times:
    1) the whole forward backward propagation
    2) the forward and loss calculation after each epoch. Hence the argument: compute_loss
    """

    ##forward pass
    o,k,h = [0]*2, [0]*3,[0]*3
    
    #linear forward (1)
    for j in range(3):
        for i in range(2):
            k[j] += W[i][j] * X[i]
        k[j] += b[j]
        
    #sigmoid forward
    for i in range(3): h[i] =  sigmoid(k[i])
        
    #linear forward (2)
    for i in range(2):
        for j in range(3): 
            o[i] += h[j] * V[j][i]
        o[i] += c[i]
        
    #softmax forward (probs)
    #print(o)
    sum_output = sum([math.exp(output) for output in o ])
    probs = [math.exp(output) / sum_output for output in o]

    #prediction for accuracy
    pred = probs.index(max(probs))

    #for computing the loss over epoch, only called when computing the loss after each epoch
    if compute_loss: return(pred,loss_forward(probs.copy(), y))
    
    ##backward
    dprobs, do =  [[0,0],[0,0]] , [0,0]
    #loss backward
    doutputs = loss_backward(probs.copy(), y)

    dV, dc, dh = [[0,0],[0,0], [0,0]], [0,0], [0,0,0]
    dh = [0,0,0]
    for i in range(2):
        if i == y: do[i] = probs[i] -1
        else: do[i] =  probs[i]  # "- 0"
    
    #gradient of V and c (2)
    for i in range(2):
        for j in range(3):
            dV[j][i] = do[i]*h[j]
            dh[j] += do[i]*V[j][i]
        dc[i] = do[i]
    dk = [None]*3
    
    #backward sigmoid
    for i in range(3): dk[i] = dh[i]*h[i]*(1-h[i])
    dW, db = [[0,0,0],[0,0,0]], [0,0,0]
    
    #gradient of W and b (1)
    for j in range(3):
        for i in range(2):
            dW[i][j] = dk[j]*X[i]
        db[j] += dk[j]     

    #W, b
    for j in range(3):
        for i in range(2):
            W[i][j] = W[i][j] -lr*dW[i][j]
        b[j] += -lr*db[j]
    #V, c
    for i in range(2):
        for j in range(3):
            V[j][i] = V[j][i] -lr*dV[j][i]
        c[i] += -lr*dc[i]

    if print_params: # for Q2
        print("dW: ", dW)
        print("\ndb: ", db)
        print("\ndV: ", dV)
        print("\ndc: ", dc)
        print("\nh: ",  h)

    return(W,b,V,c)

"""
Question 2
1) initialize weights
2) make an instance, according to the assigment
3) do a forward and backward pass

"""

W,V,b,c = initialize_weights()
X, y = [1,-1], 0
Q2 = forward_backward(X, y, W, V, b, c, print_params = True)


"""
Question 3: load in the synth data and a loop for your network.
1) load_synth function from from https://gist.github.com/pbloem/bd8348d58251872d9ca10de4816945e4
2) training loop
3) plot progress of validation data and the trainingdata (after each epoch, not during)
"""
#1)
import numpy as np
from urllib import request
import gzip
import pickle
import os

def load_synth(num_train=60_000, num_val=10_000):
    """
    Load some very basic synthetic data that should be easy to classify. Two features, so that we can plot the
    decision boundary (which is an ellipse in the feature space).
    :param num_train: Number of training instances
    :param num_val: Number of test/validation instances
    :param num_features: Number of features per instance
    :return: Two tuples (xtrain, ytrain), (xval, yval) the training data is a floating point numpy array:
    """

    THRESHOLD = 0.6
    quad = np.asarray([[1, 0.5], [1, .2]])

    ntotal = num_train + num_val

    x = np.random.randn(ntotal, 2)

    # compute the quadratic form
    q = np.einsum('bf, fk, bk -> b', x, quad, x)
    y = (q > THRESHOLD).astype(np.int)

    return (x[:num_train, :], y[:num_train]), (x[num_train:, :], y[num_train:]), 2


#load data    
train, val, num_features = load_synth()
X_train , y_train = train
X_val, y_val = val

#2)
W,V,b,c = initialize_weights(random=True, type_="normal")
epochs = 30
dV, dc, dh = [[0,0],[0,0], [0,0]], [0,0], [0,0,0]
dW, db = [[0,0,0],[0,0,0]], [0,0,0]

"""
The main training loop for Q2.
"""

# make some list to collect losses etc
ave_loss_train, ave_loss_val = [], []
accuracy_train, accuracy_val = [] , []
#decay/learningrate (decay is not used in report)
decay = 1
learningrate = 0.01
for epoch in range(epochs):
    # make some list to collect the losses etc
    loss_train, loss_val = [], []
    acc_train , acc_val =[], []

    # forward and backward
    for observation, target in zip(X_train, y_train):
        W,b,V,c= forward_backward(list(observation), target,
                                                W, V, b, c,  lr = learningrate)
    learningrate = learningrate*decay
    
    #loss after whole epoch (train data)
    for observation, target in zip(X_train, y_train):
        tpred, ltrain = forward_backward(list(observation), target, 
                                                W, V, b, c,  compute_loss = True)
        loss_train.append(ltrain)
        acc_train.append(tpred == target)

    #loss after whole epoch (validation data)
    for observation, target in zip(X_val, y_val):
        vpred, lval = forward_backward(list(observation), target,
                                                W, V, b, c, compute_loss = True)
        loss_val.append(lval)
        acc_val.append(vpred == target)

    #losses
    ave_loss_train.append(sum(loss_train)/X_train.shape[0]) 
    ave_loss_val.append(sum(loss_val)/X_val.shape[0])

    #accuracies
    accuracy_train.append(sum(acc_train)/X_train.shape[0])
    accuracy_val.append(sum(acc_val)/X_val.shape[0])

    print("Epoch: ", epoch+1)
    print("Train loss: {} --- Validation loss {} \nAccuracy train {} --- Accuracy validation {} \n ".format( 
        round(ave_loss_train[epoch], 4), round(ave_loss_val[epoch],4), round(accuracy_train[epoch],2),
        round(accuracy_val[epoch],2)))

## Figure for losses and accuracies
fig = make_subplots(1,2, horizontal_spacing=0.15)
x = list(range(1,epochs+1))
fig.add_trace(go.Scatter(x = x , #p = loss, y = accuracies
                        y = ave_loss_train,
                        name = "Training data",
                        connectgaps = True,
                        line_color = 'blue'),
                        row=1, col=1)
fig.add_trace(go.Scatter(x = x , #p = loss, y = accuracies
                        y = ave_loss_val,
                        name = "Validation data",
                        connectgaps = True,
                        line_color = 'red'),
                        row=1, col=1)
fig.add_trace(go.Scatter(x = x , #p = loss, y = accuracies
                        y = accuracy_train,
                        name = "Training data",
                        connectgaps = True,
                        line_color = 'blue',
                        showlegend=False
                        ),
                        row=1, col=2, )
fig.add_trace(go.Scatter(x = x , #p = loss, y = accuracies
                        y = accuracy_val,
                        name = "Validation data",
                        connectgaps = True,
                        line_color = 'red',
                        showlegend=False),
                        row=1, col=2)

fig.update_xaxes(title_text = "Epoch", row = 1, col = 1)
fig.update_yaxes(title_text = "Average loss (after epoch)", row =1, col =1)
fig.update_xaxes(title_text = "Epoch", row = 1, col = 2)
fig.update_yaxes(title_text = "Accuracy", row =1, col =2)
fig.update_xaxes(nticks = 20)
fig.update_layout(
    title = "The validation, training loss and accuracy on the synthetic data",
    title_x=0.5,
    height=450, width=800,
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    ),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5
)
)
fig

"""
Next up are the classes used for answering the fourth and fifth question and extra analysis.
There are a few classes.
"""

import numpy as np

class Layer():
    """
    The Layer class initializes a linear forward layer.
    It holds several objects such as the input, weights, derivatives etc.
    It also has a forward and backward function
    """
    def __init__(self, connections, neurons, initializer = "normal"):
        self.neurons = neurons

        if initializer == "normal":
            self.weights = np.random.normal(0, 1, size=(connections, neurons)) 
        elif initializer == "glorot_normal": #tried implementing https://visualstudiomagazine.com/articles/2019/09/05/neural-network-glorot.aspx
            term = np.sqrt(2.0/(connections+neurons))
            self.weights = np.random.normal(0, term, size=(connections, neurons))

        self.bias = np.zeros((1, neurons))
        self.name = "layer"
    
    def forward(self, X = None):
        self.X = X
        self.values = np.dot(X,self.weights) + self.bias

    def backward(self, grad):
        self.dweights = self.X.T @ grad                         
        self.dbias = np.sum(grad, axis = 0,keepdims=True)
        self.dvalues =  grad @ self.weights.T

class Activation:
    """
    The Activation class initialises an activation (hidden and output).
    It holds several objects such as the input, weights, derivatives etc.
    It also has a forward and backward function.
    I tried to experiment with w few activation functions from: https://en.wikipedia.org/wiki/Activation_function
    """
    def __init__(self, spec = None):
        self.spec = spec
        self.name = "activation"

    def forward(self, X):
        self.X = X
        if self.spec == "softmax":   self.values = np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
        elif self.spec == "sigmoid": self.values = 1/(1+np.exp(-X))
        elif self.spec == "relu":    self.values = np.maximum(0, X)
        elif self.spec == "tanh":    self.values = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
        
    def backward(self,grad):
        if self.spec == "softmax":   self.dvalues = grad
        elif self.spec == "sigmoid": self.dvalues = grad * self.values * (1-self.values)
        elif self.spec == "relu": 
            grad[self.X<=0] = 0
            self.dvalues =  grad
        elif self.spec == "tanh": self.dvalues = grad * (1-self.values * self.values)

class Loss:
    """
    The Loss class initialises a loss function.
    For binary classification I also use the crossentropy but was playing around with a seperate function.
    It holds several objects such as the input, weights, derivatives etc.
    It also has a forward and backward function
    """
    def __init__(self, spec = None, classes = None):
        self.spec = spec
        self.name = "loss"
        self.classes = classes

    def forward(self, X, y):
        if self.spec == "crossentropy":  loss = np.sum(-np.log(X[range(X.shape[0]), y])) / X.shape[0]
        # elif self.spec == "binary_entropy": loss = -1/len(y) * np.sum(np.log(X[:,y]))
        return(loss)

    def backward(self, X, y):
        # if self.spec == "binary_entropy":
        #     X[range(len(y)), y] -= 1 
        #     self.dloss = X / len(y)

        if self.spec == "crossentropy":
            X[range(len(y)), y] -= 1 
            self.dloss = X

class Optimizer:
    """
    The Optimizer class initialises the optimizer (only sgd).
    It updates the weights of the Layer class after forward
    """
    def __init__(self, spec = None, learningrate = 0.01):
        self.spec = spec
        self.learningrate = learningrate
        self.name = "optimizer"
        
    def update_parameters(self, layer):
        if self.spec == "sgd":
            layer.weights = layer.weights - self.learningrate* layer.dweights
            layer.bias = layer.bias - self.learningrate* layer.dbias

class Network:
    """
    The Network class is the class where it al comes together.
    It adds layers, activations together. 
    It is provided with a training and predict function.
    It is fast since it can use batches. 
    """
    def __init__(self):
        self.layers = []
        self.aes = []

    def add(self,layer):
        if layer.name == "activation" or layer.name == "layer": self.layers.append(layer)
        elif layer.name == "optimizer": self.optimizer =  layer
        elif layer.name == "loss":      self.loss = layer

    def train(self,X, y, X_val, y_val, epochs, learningrate = 0.01, batchsize = 128, eval_batch = False):  
        self.optimizer.learningrate = learningrate
        num_batches = round(X.shape[0]/batchsize)
        self.loss.classes = len(set(y))
        batches, y_batches = np.array_split(X, num_batches), np.array_split(y, num_batches)
        losses, losses_val, batch_loss = [], [],[]
        accuracies, accuracies_val, batch_acc = [], [] ,[]

        for epoch in range(0,epochs):

            print("Epoch: " + str(epoch+1))
            for batch, y_batch in zip(batches, y_batches):
                #forward
                for index, layer in enumerate(self.layers):
                    if index == 0: layer.forward(batch)
                    elif index != 0: layer.forward(self.layers[index -1].values)   

                self.loss.backward(self.layers[len(self.layers) -1].values,y_batch)

                #backward
                for index, layer in enumerate(self.layers[::-1]):
                    if index == 0:    layer.backward(self.loss.dloss)
                    elif index != 0:  layer.backward(self.layers[len(self.layers) - (index)].dvalues)
                
                #update params
                for index, layer in enumerate(self.layers):
                    if layer.name == "layer": self.optimizer.update_parameters(layer)
                
                batch_l, batch_a = self.predict(batch, y_batch)
                batch_loss.append(batch_l)
                batch_acc.append(batch_a)

            loss, acc = self.predict(X,y)
            loss_val , acc_val = self.predict(X_val,y_val)


            losses.append(loss)
            accuracies.append(acc)
            losses_val.append(loss_val)
            accuracies_val.append(acc_val)

            print("Train loss: {} --- Validation loss {} \nAccuracy train {} --- Accuracy validation {} \n ".format( 
                                        round(losses[epoch], 4), 
                                        round(losses_val[epoch],4), 
                                        round(accuracies[epoch],4),
                                        round(accuracies_val[epoch],4)))

        if eval_batch:
            return(losses, losses_val, accuracies, accuracies_val, batch_acc, batch_loss)
        else:
            return(losses, losses_val, accuracies, accuracies_val)

    def predict(self,X, y, eval =  True):
        for index, layer in enumerate(self.layers):
            if index == 0: layer.forward(X)
            elif index != 0: layer.forward(self.layers[index -1].values) 

        if eval: 
            preds = list(np.argmax(self.layers[len(self.layers) -1].values, axis=1))
            accuracy = np.mean(preds == y)
            loss = self.loss.forward(self.layers[len(self.layers) -1].values,y)
            return(loss, accuracy)

        return(self.layers[len(self.layers) -1].values)

"""
The next code is for loading the datasets.
"""

# From https://gist.github.com/pbloem/bd8348d58251872d9ca10de4816945e4

def load_mnist(final=False, flatten=True):
    """
    Load the MNIST data
    :param final: If true, return the canonical test/train split. If false, split some validation data from the training
       data and keep the test data hidden.
    :param flatten:
    :return:
    """

    if not os.path.isfile('mnist.pkl'):
        init()

    xtrain, ytrain, xtest, ytest = load()
    xtl, xsl = xtrain.shape[0], xtest.shape[0]

    if flatten:
        xtrain = xtrain.reshape(xtl, -1)
        xtest  = xtest.reshape(xsl, -1)

    if not final: # return the flattened images
        return (xtrain[:-5000], ytrain[:-5000]), (xtrain[-5000:], ytrain[-5000:]), 10

    return (xtrain, ytrain), (xtest, ytest), 10

# Numpy-only MNIST loader. Courtesy of Hyeonseok Jung
# https://github.com/hsjeong5/MNIST-for-Numpy
filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]
def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

"""
Question 4: implement the network for MNIST. Input 784 -> L(784, 300) -> Sig() -> L(300, 10) -> Softmax

1) Using the classes stated above one can create this network.
"""

mod = Network()
mod.add(Layer(784,300))
mod.add(Activation("sigmoid"))
mod.add(Layer(300,10))
mod.add(Activation("softmax"))
mod.add(Loss("crossentropy"))
mod.add(Optimizer("sgd"))



"""
Question 5: Analyses of the MNIST data with the network
1) I will use the dataset where final = False, this is used to estimate the hyperparams
2) I will use the dataset where final = True, this is the final training with the chosen hyperparams
"""
# load in data
train, val, nof = load_mnist()
X_train, y_train = train
X_val, y_val = val

#standardize data based on mu and sigma
mu , sigma = np.mean(X_train), np.std(X_train)
X_train =(X_train-mu)/(sigma)
X_val = (X_val-mu)/(sigma)

# train the model (test if it works)
losses, losses_val, accuracies, accuracies_val = mod.train(X_train, y_train,
                               X_val, y_val,
                                  epochs = 5, batchsize = 512,
                                  learningrate= 0.01)

"""
First I will determine a good learningrate for the batchsizes: 64, 128, 256, 512 and 1024
Results are shown in table below, I reused the code everytime since I am only interested for the learningrate. 
Eventually opted for batch of 2048.
Batchsize         :  64  |  128  |  256  |  512  |  1024
"Best learningrate" :  0.01|  0.01 | 0.01 |  0.01 | 0.01
"""
mod2 = Network()
mod2.add(Layer(784,300))
mod2.add(Activation("sigmoid"))
mod2.add(Layer(300,10))
mod2.add(Activation("softmax"))
mod2.add(Loss("crossentropy"))
mod2.add(Optimizer("sgd"))

# train the model (test if it works)
losses2, losses_val2, accuracies2, accuracies_val2, batch_acc2, batch_loss2 = mod2.train(X_train, y_train,
                                                                                    X_val, y_val,
                                                                                        epochs = 10, batchsize = 2048,
                                                                                        learningrate= 0.01,
                                                                                        eval_batch= True)

# graph
step = 270
x_batch = list(np.arange(0,10, 10/step))
x = list(range(1,11))

fig2 = make_subplots(1,2, horizontal_spacing=0.15)

fig2.add_trace(go.Scatter(x = x , #p = loss, y = accuracies
                         y = losses,
                         name = "Training data",
                        connectgaps = True,
                        line_color = 'blue'),
                        row=1, col=1)
fig2.add_trace(go.Scatter(x = x , #p = loss, y = accuracies
                         y = losses_val,
                         name = "Validation data",
                        connectgaps = True,
                        line_color = 'red'),
                        row=1, col=1)
fig2.add_trace(go.Scatter(x = x_batch , #p = loss, y = accuracies
                         y = batch_acc,
                         name = "Batch (size =  2048)",
                        connectgaps = True,
                        line_color = 'grey'),
                        row=1, col=2)
fig2.add_trace(go.Scatter(x = x , #p = loss, y = accuracies
                         y = accuracies,
                        name = "Training data",
                        connectgaps = True,
                        line_color = 'blue',
                        showlegend=False
                        ),
                        row=1, col=2, )
fig2.add_trace(go.Scatter(x = x , #p = loss, y = accuracies
                         y = accuracies_val,
                         name = "Validation data",
                        connectgaps = True,
                        line_color = 'red',
                        showlegend=False),
                        row=1, col=2)


fig2.update_xaxes(title_text = "Epoch", row = 1, col = 1)
fig2.update_yaxes(title_text = "Average loss (after epoch)", row =1, col =1)
fig2.update_xaxes(title_text = "Epoch", row = 1, col = 2)
fig2.update_yaxes(title_text = "Accuracy", row =1, col =2)
fig2.update_xaxes(nticks = 20)
fig2.update_layout(
    title = "The validation/training loss and accuracy on the MNIST data",
    title_x=0.5,
    height=450, width=800,
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    ),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5
)
)
#fig2


## run the model two more times (subquestion)
mod3 = Network()
mod3.add(Layer(784,300))
mod3.add(Activation("sigmoid"))
mod3.add(Layer(300,10))
mod3.add(Activation("softmax"))
mod3.add(Loss("crossentropy"))
mod3.add(Optimizer("sgd"))

# train the model (test if it works)
losses3, losses_val3, accuracies3, accuracies_val3, batch_acc3, batch_loss3 = mod3.train(X_train, y_train,
                                                                                    X_val, y_val,
                                                                                        epochs = 10, batchsize = 2048,
                                                                                        learningrate= 0.01,
                                                                                        eval_batch= True)
mod4 = Network()
mod4.add(Layer(784,300))
mod4.add(Activation("sigmoid"))
mod4.add(Layer(300,10))
mod4.add(Activation("softmax"))
mod4.add(Loss("crossentropy"))
mod4.add(Optimizer("sgd"))

# train the model (test if it works)
losses4, losses_val4, accuracies4, accuracies_val4, batch_acc4, batch_loss4 = mod4.train(X_train, y_train,
                                                                                    X_val, y_val,
                                                                                        epochs = 10, batchsize = 2048,
                                                                                        learningrate= 0.01,
                                                                                        eval_batch= True)

""""
Now, I have three times each list (train, val, batch; accuracy, loss).
1)compute mean and stdev of each iteration
2) plot the lines (long code)
This could be done with less and probably easier code.
""""

losses_mean_train, losses_mean_val = [],[]
losses_sd_train, losses_sd_val = [],[]
max_train_loss, min_train_loss = [], []
max_val_loss, min_val_loss = [],[]

accuracy_mean_train, accuracy_mean_val = [], []
accuracy_sd_train, accuracy_sd_val = [], []
max_train_acc, min_train_acc = [],[]
max_val_acc, min_val_acc = [],[]

accuracy_mean_batch, accuracy_sd_batch = [], []
min_batch, max_batch = [],[]


for i in range(len(losses2)):
    #mean and sd losses
    l_train = [losses2[i],losses3[i],losses4[i]]
    l_val = [losses_val2[i],losses_val3[i],losses_val4[i]]

    losses_mean_train.append(np.mean(l_train))
    losses_mean_val.append(np.mean(l_val))

    losses_sd_train.append(np.std(l_train))
    losses_sd_val.append(np.std(l_val))

    min_train_loss.append(losses_mean_train[i] - losses_sd_train[i])
    min_val_loss.append(losses_mean_val[i] - losses_sd_val[i])

    max_train_loss.append(losses_mean_train[i] +losses_sd_train[i] )
    max_val_loss.append(losses_mean_val[i]+ losses_sd_val[i])

    #mean and sd accuracy
    a_train = [accuracies2[i],accuracies3[i],accuracies4[i]]
    a_val = [accuracies_val2[i],accuracies_val3[i],accuracies_val4[i]]

    accuracy_mean_train.append(np.mean(a_train))
    accuracy_mean_val.append(np.mean(a_val))

    accuracy_sd_train.append(np.std(a_train))
    accuracy_sd_val.append(np.std(a_val))

    min_train_acc.append(accuracy_mean_train[i] - accuracy_sd_train[i])
    min_val_acc.append(accuracy_mean_val[i] - accuracy_sd_val[i])

    max_train_acc.append(accuracy_mean_train[i] +accuracy_sd_train[i] )
    max_val_acc.append(accuracy_mean_val[i]+ accuracy_sd_val[i])

for j in range(len(batch_acc2)):
    a_batch = [batch_acc4[j], batch_acc3[j], batch_acc2[j]]
    accuracy_mean_batch.append(np.mean(a_batch))
    accuracy_sd_batch.append(np.std(a_batch))

    min_batch.append(accuracy_mean_batch[j] - accuracy_sd_batch[j])
    max_batch.append(accuracy_mean_batch[j] + accuracy_sd_batch[j])

# graph
x = list(range(1,11))

fig3 = make_subplots(1,2, horizontal_spacing=0.15)
step = 270
x_batch = list(np.arange(0,10, 10/step))

fig3.add_trace(go.Scatter(x = x_batch ,
                         y = min_batch,
                        name = "Training data",
                        connectgaps = True,
                        fill = None,
                        showlegend=False,
                        line_width = 0,
                        mode= "lines",
                        line_color = "grey"
                        ),
                        row=1, col=2, )
fig3.add_trace(go.Scatter(x = x_batch , 
                         y = max_batch,
                         name = "Validation data",
                        connectgaps = True,
                        showlegend=False,
                        line_width = 0,
                        mode= "lines",
                        line_color = "grey",
                        fill="tonexty"),
                        row=1, col=2)

fig3.add_trace(go.Scatter(x = x_batch , 
                         y = accuracy_mean_batch,
                         name = "Batch (size =  2048)",
                        connectgaps = True,
                        line_color = 'black',
                        line_width = 1),
                        row=1, col=2)
fig3.add_trace(go.Scatter(x = x , 
                         y = min_train_acc,
                        name = "Training data",
                        mode="lines",
                        connectgaps = True,
                        showlegend=False,
                        fill=None,
                        line_width = 0,
                        line_color = 'blue'),
                        row=1, col=1)
fig3.add_trace(go.Scatter(x = x , 
                         y = max_train_acc,
                         name = "Validation data",
                        connectgaps = True,
                        showlegend=False,
                        line_width = 0,
                        mode= "lines",
                        fill="tonexty",
                        line_color = 'blue'),
                        row=1, col=1)

fig3.add_trace(go.Scatter(x = x , 
                         y = min_val_acc,
                        name = "Training data",
                        mode="lines",
                        connectgaps = True,
                        showlegend=False,
                        fill=None,
                        line_width = 0,
                        line_color = 'red'),
                        row=1, col=1)

fig3.add_trace(go.Scatter(x = x , 
                         y = max_val_acc,
                         name = "Validation data",
                        connectgaps = True,
                        showlegend=False,
                        line_width = 0,
                        mode= "lines",
                        fill="tonexty",
                        line_color = 'red'),
                        row=1, col=1)

fig3.add_trace(go.Scatter(x = x , 
                         y = accuracy_mean_train,
                         name = "Validation data",
                        connectgaps = True,
                        line_width = 1,
                        mode= "lines",
                        line_color = 'blue'),
                        row=1, col=1)

fig3.add_trace(go.Scatter(x = x , 
                         y = accuracy_mean_val,
                         name = "Validation data",
                        connectgaps = True,
                        line_width = 1,
                        mode= "lines",
                        line_color = 'red'),
                        row=1, col=1)



fig3.update_xaxes(title_text = "Epoch", row = 1, col = 1)
fig3.update_yaxes(title_text = "Accuracy", range=[0.0, 1.0], row =1, col =1)
fig3.update_xaxes(title_text = "Epoch", range = [0,10] ,row = 1, col = 2)
fig3.update_yaxes(title_text = "Accuracy", range=[0.0, 1.0],row =1, col =2)
fig3.update_xaxes(nticks = 20, row=1, col = 2)
fig3.update_layout(
    title = "The validation, training and batch accuracy on the MNIST data",
    title_x=0.5,
    height=450, width=800,
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    ),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5
)
)
fig3



"""
Now we train the model on different learning rates to understand how the learningrate influences
the learning of the model.
"""
mod5 = Network()
mod5.add(Layer(784,300))
mod5.add(Activation("sigmoid"))
mod5.add(Layer(300,10))
mod5.add(Activation("softmax"))
mod5.add(Loss("crossentropy"))
mod5.add(Optimizer("sgd"))

# train the model (test if it works)
losses5, losses_val5, accuracies5, accuracies_val5, batch_acc5, batch_loss5 = mod5.train(X_train, y_train,
                                                                                    X_val, y_val,
                                                                                        epochs = 10, batchsize = 2048,
                                                                                        learningrate= 0.1,
                                                                                        eval_batch= True)

mod6 = Network()
mod6.add(Layer(784,300))
mod6.add(Activation("sigmoid"))
mod6.add(Layer(300,10))
mod6.add(Activation("softmax"))
mod6.add(Loss("crossentropy"))
mod6.add(Optimizer("sgd"))

# train the model (test if it works)
losses6, losses_val6, accuracies6, accuracies_val6, batch_acc6, batch_loss6 = mod6.train(X_train, y_train,
                                                                                    X_val, y_val,
                                                                                        epochs = 10, batchsize = 2048,
                                                                                        learningrate= 0.01,
                                                                                        eval_batch= True)

mod7 = Network()
mod7.add(Layer(784,300))
mod7.add(Activation("sigmoid"))
mod7.add(Layer(300,10))
mod7.add(Activation("softmax"))
mod7.add(Loss("crossentropy"))
mod7.add(Optimizer("sgd"))

# train the model (test if it works)
losses7, losses_val7, accuracies7, accuracies_val7, batch_acc7, batch_loss7 = mod7.train(X_train, y_train,
                                                                                    X_val, y_val,
                                                                                        epochs = 10, batchsize = 2048,
                                                                                        learningrate= 0.001,
                                                                                        eval_batch= True)
                                                            
mod8 = Network()
mod8.add(Layer(784,300))
mod8.add(Activation("sigmoid"))
mod8.add(Layer(300,10))
mod8.add(Activation("softmax"))
mod8.add(Loss("crossentropy"))
mod8.add(Optimizer("sgd"))

# train the model (test if it works)
losses8, losses_val8, accuracies8, accuracies_val8, batch_acc8, batch_loss8 = mod8.train(X_train, y_train,
                                                                                    X_val, y_val,
                                                                                        epochs = 10, batchsize = 2048,
                                                                                        learningrate= 0.0001,
                                                                                        eval_batch= True)


mod9 = Network()
mod9.add(Layer(784,300))
mod9.add(Activation("sigmoid"))
mod9.add(Layer(300,10))
mod9.add(Activation("softmax"))
mod9.add(Loss("crossentropy"))
mod9.add(Optimizer("sgd"))

# train the model (test if it works)
losses9, losses_val9, accuracies9, accuracies_val9, batch_acc9, batch_loss9 = mod9.train(X_train, y_train,
                                                                                    X_val, y_val,
                                                                                        epochs = 10, batchsize = 2048,
                                                                                        learningrate= 0.00001,
                                                                                        eval_batch= True)




fig4 = go.Figure()
fig4.add_trace(go.Scatter(x = x_batch ,
                         y = batch_acc5,
                         name = "Learningrate: 0.1",
                        connectgaps = True,
                        line_color = 'olive'))
fig4.add_trace(go.Scatter(x = x_batch ,
                         y = batch_acc6,
                         name = "Learningrate: 0.01",
                        connectgaps = True,
                        line_color = 'grey'))
fig4.add_trace(go.Scatter(x = x_batch , 
                         y = batch_acc7,
                         name = "Learningrate: 0.001",
                        connectgaps = True,
                        line_color = 'midnightblue'))
fig4.add_trace(go.Scatter(x = x_batch ,
                         y = batch_acc8,
                        name = "Learningrate: 0.0001",
                        connectgaps = True,
                        line_color = 'firebrick'
                        ))
fig4.add_trace(go.Scatter(x = x_batch ,
                         y = batch_acc9,
                         name = "Learningrate: 0.00001",
                        connectgaps = True,
                        line_color = 'maroon'))



fig4.update_xaxes(title_text = "Epoch")
fig4.update_yaxes(title_text = "Accuracy")

fig4.update_xaxes(nticks = 10)
fig4.update_layout(
    title = "The batch accuracy on the MNIST data for different learningrates",
    title_x=0.5,
    height=450, width=800,
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    )
)
# fig
# fig2
# fig3
# fig4

"""
Now I initialize the best model (mod5) again and train it on the full data and eventually test it.
Steps:
1) load in data
2) standardize data
3) train
4) evaluate on test set (only last epoch counts since one cannot make any model decisions based on testdata)

"""
import pickle
# data
train, test, nof = load_mnist(final=True)
X_train, y_train = train
X_test, y_test = test

#standardize data based on mu and sigma
mu , sigma = np.mean(X_train), np.std(X_train)
X_train =(X_train-mu)/(sigma)
X_test = (X_test-mu)/(sigma)

X_train.shape


#make model
mod_final = Network()
mod_final.add(Layer(784,300))
mod_final.add(Activation("sigmoid"))
mod_final.add(Layer(300,10))
mod_final.add(Activation("softmax"))
mod_final.add(Loss("crossentropy"))
mod_final.add(Optimizer("sgd"))

# train the model (test if it works)
losses_final, losses_val_final, accuracies_final, accuracies_val_final, batch_acc_final, batch_loss_final, = mod_final.train(X_train, y_train,
                                                                                    X_test, y_test,
                                                                                        epochs = 5, batchsize = 2048,
                                                                                        learningrate= 0.01,
                                                                                        eval_batch= True)


print("Final train loss: {} --- Final test loss: {} \nFinal train accuracy: {} --- Final accuracy test {} \n ".format( 
                            round(losses_final[4], 4), 
                            round(losses_val_final[4],4), 
                            round(accuracies_final[4],4),
                            round(accuracies_val_final[4],4)))
# graph
step = 5/125
x_batch = list(np.arange(0,5, step))
x = list(range(1,6))
fig_final = make_subplots(1,2, horizontal_spacing=0.15)

fig_final.add_trace(go.Scatter(x = x , 
                         y = losses_final,
                         name = "Training data",
                        connectgaps = True,
                        line_color = 'blue'),
                        row=1, col=1)

fig_final.add_trace(go.Scatter(x = x_batch , 
                         y = batch_acc_final,
                         name = "Batch (size =  2048)",
                        connectgaps = True,
                        line_color = 'grey'),
                        row=1, col=2)
fig_final.add_trace(go.Scatter(x = x , 
                         y = accuracies_final,
                        name = "Training data",
                        connectgaps = True,
                        line_color = 'blue',
                        showlegend=False
                        ),
                        row=1, col=2, )


fig_final.update_xaxes(title_text = "Epoch", row = 1, col = 1)
fig_final.update_yaxes(title_text = "Average loss (after epoch)", row =1, col =1)
fig_final.update_xaxes(title_text = "Epoch", row = 1, col = 2)
fig_final.update_yaxes(title_text = "Accuracy", row =1, col =2)
fig_final.update_xaxes(nticks = 9)
fig_final.update_layout(
    title = "The training loss and accuracy of the final run on the MNIST data",
    title_x=0.5,
    height=450, width=800,
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    ),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5
)
)



################# extra ##################
"""
It is interesting to experiment with the activation functions and extra layers.
"""
#from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py


"""
code below is for loading data set, from : https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
"""
def load_mnist_fashion(path = "C:\\Users\\Jens\\Desktop\\Master Business Analytics\\Deep Learning\\fashion", kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

# import original training data and split in validation and training
images, labels = load_mnist_fashion()
X_train_f, X_val_f = images[:-5000], images[-5000:]
y_train_f, y_val_f = labels[:-5000], labels[-5000:]

#mu and sigma
mu_f = np.mean(X_train_f)
sigma_f = np.std(X_train_f)

X_train_f, X_val_f = (X_train_f - mu_f)/sigma_f, (X_val_f -mu_f)/sigma_f

#number of labels
print(len(set(y_train)))


"""
First I tried the model that worked fine on the normal MNIST dataset.
Not bad, but isn't great either
"""
modf = Network()
modf.add(Layer(784,300))
modf.add(Activation("sigmoid"))
modf.add(Layer(300,10))
modf.add(Activation("softmax"))
modf.add(Loss("crossentropy"))
modf.add(Optimizer("sgd"))

#acc val = 0.7868, acc train = 0.8037
lossesf, losses_valf, accuraciesf, accuracies_valf, batch_accf, batch_lossf = modf.train(X_train_f, y_train_f,
                                                                                        X_val_f, y_val_f,
                                                                                        epochs = 10, batchsize = 2048,
                                                                                        learningrate= 0.01,
                                                                                        eval_batch= True)

"""
The second model I used is the same except for the number of params 
"""
modf2 = Network()
modf2.add(Layer(784,512))
modf2.add(Activation("sigmoid"))
modf2.add(Layer(512,10))
modf2.add(Activation("softmax"))
modf2.add(Loss("crossentropy"))
modf2.add(Optimizer("sgd"))

lossesf2, losses_valf2, accuraciesf2, accuracies_valf2, batch_accf2, batch_lossf2 = modf2.train(X_train_f, y_train_f,
                                                                                        X_val_f, y_val_f,
                                                                                        epochs = 10, batchsize = 2048,
                                                                                        learningrate= 0.001,
                                                                                        eval_batch= True)
"""
In the third model I tried the tanh and glorot uniform for the weights.
also used some different activations
"""
modf3 = Network()
modf3.add(Layer(784,512,initializer = "glorot_normal"))
modf3.add(Activation("sigmoid"))
modf3.add(Layer(512,256,initializer = "glorot_normal"))
modf3.add(Activation("tanh"))
modf3.add(Layer(256,128,initializer = "glorot_normal"))
modf3.add(Activation("relu"))
modf3.add(Layer(128,10,initializer = "glorot_normal"))
modf3.add(Activation("softmax"))
modf3.add(Loss("crossentropy"))
modf3.add(Optimizer("sgd"))

#acc val = 0.6804, acc train = 0.7447
lossesf3, losses_valf3, accuraciesf3, accuracies_valf3, batch_accf3, batch_lossf3 = modf3.train(X_train_f, y_train_f,
                                                                                        X_val_f, y_val_f,
                                                                                        epochs = 10, batchsize = 2048,
                                                                                        learningrate= 0.0001,
                                                                                        eval_batch= True)
"""
In the fourth model I tried the Relu. Results in nan loss and 1/10 acc. 
"""
modf4 = Network()
modf4.add(Layer(784,512))
modf4.add(Activation("sigmoid"))
modf4.add(Layer(512,256))
modf4.add(Activation("tanh"))
modf4.add(Layer(256,10))
modf4.add(Activation("softmax"))
modf4.add(Loss("crossentropy"))
modf4.add(Optimizer("sgd"))

#acc val = 0.6804, acc train = 0.7447
lossesf4, losses_valf4, accuraciesf4, accuracies_valf4, batch_accf4, batch_lossf4 = modf4.train(X_train_f, y_train_f,
                                                                                        X_val_f, y_val_f,
                                                                                        epochs = 10, batchsize = 2048,
                                                                                        learningrate= 0.0001,
                                                                                        eval_batch= True)


"""
To see if glorot works better
"""
modf5 = Network()
modf5.add(Layer(784,512,initializer = "glorot_normal"))
modf5.add(Activation("sigmoid"))
modf5.add(Layer(512,256,initializer = "glorot_normal"))
modf5.add(Activation("tanh"))
modf5.add(Layer(256,10, initializer = "glorot_normal"))
modf5.add(Activation("softmax"))
modf5.add(Loss("crossentropy"))
modf5.add(Optimizer("sgd"))

#acc val = 0.6804, acc train = 0.7447
lossesf5, losses_valf5, accuraciesf5, accuracies_valf5, batch_accf5, batch_lossf5 = modf5.train(X_train_f, y_train_f,
                                                                                        X_val_f, y_val_f,
                                                                                        epochs = 10, batchsize = 2048,
                                                                                        learningrate= 0.0001,
                                                                                        eval_batch= True)

"""
I decided to leave out the first model, it was not that interesting.

"""
fig5 = make_subplots(1,2, horizontal_spacing=0.15)
                        
fig5.add_trace(go.Scatter(x = x_batch ,
                         y = batch_accf2,
                         name = "Model 1",
                        connectgaps = True,
                        line_color = 'firebrick'),
                        row=1, col=1)
fig5.add_trace(go.Scatter(x = x_batch , 
                         y = batch_accf3,
                         name = "Model 2",
                        connectgaps = True,
                        line_color = 'midnightblue'),
                        row=1, col=1)
fig5.add_trace(go.Scatter(x = x_batch ,
                         y = batch_accf4,
                        name = "Model 3",
                        connectgaps = True,
                        line_color = 'grey'
                        ),
                        row=1, col=1)
fig5.add_trace(go.Scatter(x = x_batch ,
                         y = batch_accf5,
                         name = "Model 4",
                        connectgaps = True,
                        line_color = 'olive'),
                        row=1, col=1)

fig5.add_trace(go.Scatter(x = x_batch ,
                         y = batch_accf2,
                         name = "Model 1",
                        connectgaps = True, showlegend=False,
                        line_color = 'firebrick'),
                        row=1, col=2)
fig5.add_trace(go.Scatter(x = x_batch , 
                         y = batch_accf3,
                         name = "Model 2",
                        connectgaps = True,showlegend=False,
                        line_color = 'midnightblue'),
                        row=1, col=2)
fig5.add_trace(go.Scatter(x = x_batch ,
                         y = batch_accf4,
                        name = "Model 3",
                        connectgaps = True,showlegend=False,
                        line_color = 'grey'
                        ),
                        row=1, col=2)
fig5.add_trace(go.Scatter(x = x_batch ,
                         y = batch_accf5,
                         name = "Model 4",
                        connectgaps = True,showlegend=False,
                        line_color = 'olive'),
                        row=1, col=2)

# fig5.update_xaxes(title_text = "Epoch")
# fig5.update_yaxes(title_text = "Accuracy")

fig5.update_xaxes(title_text = "Epoch", row = 1, col = 1)
fig5.update_yaxes(title_text = "Accuracy", range=[0.0, 1.0], row =1, col =1)
fig5.update_xaxes(title_text = "Epoch", range = [0,10] ,row = 1, col = 2)
fig5.update_yaxes(title_text = "Accuracy", range=[0.0, 1.0],row =1, col =2)

fig5.update_xaxes(nticks = 10)
fig5.update_layout(
    title = "The batch accuracy on the MNIST-fashion data for different models",
    title_x=0.5,
    height=450, width=800,
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    )
)
fig5


fig6 = make_subplots(1,2, horizontal_spacing=0.15)
x = list(range(1,epochs+1))
fig6.add_trace(go.Scatter(x = x , #p = loss, y = accuracies
                         y = lossesf5,
                         name = "Training data",
                        connectgaps = True,
                        line_color = 'blue'),
                        row=1, col=1)
fig6.add_trace(go.Scatter(x = x , #p = loss, y = accuracies
                         y = losses_valf5,
                         name = "Validation data",
                        connectgaps = True,
                        line_color = 'red'),
                        row=1, col=1)
fig6.add_trace(go.Scatter(x = x , #p = loss, y = accuracies
                         y = accuraciesf5,
                        name = "Training data",
                        connectgaps = True,
                        line_color = 'blue',
                        showlegend=False
                        ),
                        row=1, col=2, )
fig6.add_trace(go.Scatter(x = x , #p = loss, y = accuracies
                         y = accuracies_valf5,
                         name = "Validation data",
                        connectgaps = True,
                        line_color = 'red',
                        showlegend=False),
                        row=1, col=2)

fig6.update_xaxes(title_text = "Epoch", row = 1, col = 1)
fig6.update_yaxes(title_text = "Average loss (after epoch)", row =1, col =1)
fig6.update_xaxes(title_text = "Epoch", row = 1, col = 2)
fig6.update_yaxes(title_text = "Accuracy", row =1, col =2)
fig6.update_xaxes(nticks = 20)
fig6.update_layout(
    title = "The validation, training loss and accuracy on the MNIST-fashion data",
    title_x=0.5,
    height=450, width=800,
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    ),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5
)
)
fig6


"""
Final run on the MNIST-fashion dataset.
1) load in data
2) standardize data
3) train model
4) test model
"""

# import original training data and split in validation and training
X_train, y_train = load_mnist_fashion()
X_test, y_test = load_mnist_fashion(kind="t10k")
np.max(X_train)
# X_train_f, X_val_f = images[:-5000], images[-5000:]
# y_train_f, y_val_f = labels[:-5000], labels[-5000:]

mu = np.mean(X_train)
sigma = np.std(X_train)

X_train, X_test = (X_train - mu)/sigma, (X_test -mu)/sigma

mod_f_final = Network()
mod_f_final.add(Layer(784,512,initializer = "glorot_normal"))
mod_f_final.add(Activation("sigmoid"))
mod_f_final.add(Layer(512,256,initializer = "glorot_normal"))
mod_f_final.add(Activation("tanh"))
mod_f_final.add(Layer(256,10, initializer = "glorot_normal"))
mod_f_final.add(Activation("softmax"))
mod_f_final.add(Loss("crossentropy"))
mod_f_final.add(Optimizer("sgd"))

#acc val = 0.6804, acc train = 0.7447
lossesmod_f_final, losses_valmod_f_final, accuraciesmod_f_final, accuracies_valmod_f_final, batch_accmod_f_final, batch_lossmod_f_final = mod_f_final.train(X_train, y_train,
                                                                                        X_test, y_test,
                                                                                        epochs = 10, batchsize = 2048,
                                                                                        learningrate= 0.0001,
                                                                                        eval_batch= True)


print("Final train loss: {} --- Final test loss: {} \nFinal train accuracy: {} --- Final accuracy test {} \n ".format( 
                            round(lossesmod_f_final[4], 4), 
                            round(losses_valmod_f_final[4],4), 
                            round(accuraciesmod_f_final[4],4),
                            round(accuracies_valmod_f_final[4],4)))
# graph
len(batch_accmod_f_final)
step = 10/290
x_batch = list(np.arange(0,10, step))
x = list(range(1,11))
fig_final_f = make_subplots(1,2, horizontal_spacing=0.15)

fig_final_f.add_trace(go.Scatter(x = x , 
                         y = lossesmod_f_final,
                         name = "Training data",
                        connectgaps = True,
                        line_color = 'blue'),
                        row=1, col=1)

fig_final_f.add_trace(go.Scatter(x = x_batch , 
                         y = batch_accmod_f_final,
                         name = "Batch (size =  2048)",
                        connectgaps = True,
                        line_color = 'grey'),
                        row=1, col=2)
fig_final_f.add_trace(go.Scatter(x = x , 
                         y = accuraciesmod_f_final,
                        name = "Training data",
                        connectgaps = True,
                        line_color = 'blue',
                        showlegend=False
                        ),
                        row=1, col=2, )


fig_final_f.update_xaxes(title_text = "Epoch", row = 1, col = 1)
fig_final_f.update_yaxes(title_text = "Average loss (after epoch)", row =1, col =1)
fig_final_f.update_xaxes(title_text = "Epoch", row = 1, col = 2)
fig_final_f.update_yaxes(title_text = "Accuracy", row =1, col =2)
fig_final_f.update_xaxes(nticks = 20)
fig_final_f.update_layout(
    title = "The training loss and accuracy of the final run on the MNIST-fashion data",
    title_x=0.5,
    height=450, width=800,
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    ),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5
)
)


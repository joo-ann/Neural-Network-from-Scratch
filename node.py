import numpy as np
import random
import math

class Node():
    def __init__(self, w=[], b=[], p=[], v=-1):
        self.prev = p
        self.weights = w
        self.bias = b
        self.value = v
       
    
    def sigmoid(x):
        return 1/(1+(math.e)^(-x))
    

    def run(self, inp):
        if self.prev == []:
            self.value = 1/(1+2.71828**(-inp * self.weight + self.bias))
        elif self.value == -1:
            calc = np.array()
            for i in self.prev:
                if i.value == -1:
                    i.run(_ for _ in inp)
                calc.append(1/(1+2.71828**(-i.value * self.weight + self.bias)))
            
            self.value = sum(calc)

    def add_prev(self, *node):
        self.prev.append(i for i in node)

    def get_value(self):
        return self.value
    
    def get_weights(self):
        return self.weights
    
    def get_biases(self):
        return self.bias


class Network():
    def __init__(self, *layers):
        self.network = []
        self.layers = []
        self.biases = []
        self.network.append([Node() for _ in range(layers[0])])
        for i in range(1, len(layers)):
            l = []
            for j in range(layers[i]): 
                w = [(random.randint(0, 100)/100) for _ in range(layers[i-1])]
                b = random.randint(-10, 10)
                b = np.array(b)
                p = self.network[i-1]
                l.append(Node(w, b, p))
            x = []
            y = []
            for i in l:
                x.append(i.get_weights())
                y.append(i.get_biases())
                

            self.layers.append(np.array(x))
            self.biases.append(np.array(y).reshape(-1, 1))
            self.network.append(l)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, inp):
        self.activation = []
        v = inp
        for i in range(len(self.layers)):
            v = np.matmul(self.layers[i], v) + self.biases[i]
            self.activation.append(v)
            v = self.sigmoid(v)

        return v

    def get_weights(self):
        return self.layers

    def set_weights(self, l):
        self.layers = l

    def display(self): 
        visual = []
        for i in range(len(self.network)):
            l = []
            for _ in range(len(self.network[i])):
                l.append('o')
            visual.append(l)
            
        print(*visual, sep='\n')

    def get_activation(self):
        return self.activation

    def backprop(self, inp, label):
        loss = 0.5 * (self.forward(inp) - label)**2
        gradients = []
        for i in range(len(self.layers)-1):
            gradients.append()

            
test = Network(2, 2, 1)

inp = np.array([1, 1])
print(test.forward(inp))

print(test.get_weights())

print(test.get_activation())

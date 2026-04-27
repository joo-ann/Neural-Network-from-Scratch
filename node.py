import numpy as np
import random
import math
from torchvision import datasets, transforms

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
    

    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    

    def forward(self, inp):
        self.activation = [inp]
        self.z_vals = []

        v = inp

        for i in range(len(self.layers)):
            z = np.matmul(self.layers[i], v) + self.biases[i].flatten()
            self.z_vals.append(z)

            v = self.sigmoid(z)
            self.activation.append(v)

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



    def backprop(self, inp, label, lr=0.1):
        output = self.forward(inp)

        label = np.array(label)

        delta = (output - label) * self.sigmoid_deriv(self.z_vals[-1])

        nabla_w = [None] * len(self.layers)
        nabla_b = [None] * len(self.biases)

        nabla_w[-1] = np.outer(delta, self.activation[-2])
        nabla_b[-1] = delta

        for i in range(2, len(self.layers) + 1):
            z = self.z_vals[-i]
            sp = self.sigmoid_deriv(z)

            delta = np.dot(self.layers[-i + 1].T, delta) * sp

            nabla_w[-i] = np.outer(delta, self.activation[-i - 1])
            nabla_b[-i] = delta

        for i in range(len(self.layers)):
            self.layers[i] -= lr * nabla_w[i]
            self.biases[i] -= lr * nabla_b[i].reshape(-1, 1)

        return 0.5*(output-label)**2

    def train(self, inp, label, epochs, validation_split=0.1):
        split = int(len(inp) * (1 - validation_split))

        train_x = inp[:split]
        train_y = label[:split]

        valid_x = inp[split:]
        valid_y = label[split:]

        for n in range(epochs):
            total_loss = 0

            idx = np.arange(len(train_x))
            np.random.shuffle(idx)

            for i in idx:
                loss = self.backprop(train_x[i], train_y[i])
                total_loss += loss

            val_loss = 0
            for i in range(len(valid_x)):
                output = self.forward(valid_x[i])
                val_loss += 0.5 * (output - valid_y[i])**2

            print(f"Epoch {n}: train_loss={total_loss}, val_loss={val_loss}")

    
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())

features = []
labels = []

for x, y in train_data:
    features.append(x.view(-1).tolist())
    labels.append(y)

network = Network(784, 16, 16, 10)

network.train(features, labels)

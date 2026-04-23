import numpy as np
from node import Node
import random

class Network():
    def __init__(self, *layers):
        self.network = []
        for i in range(len(layers)):
            w = [random.randint(-100, 100) for _ in range(i)]
            w = np.array(w)
            b = random.randint(-10, 10)
            l = [Node(w, b, []) for _ in range(i)]
            self.network.append(l)
        
        for i in range(1, len(self.network)):
            for j in range(len(self.network[i])):
                self.network[i][j].add_next(_ for _ in self.network[i-1])

    def dataset(self, labels, *features):
        self.labels = labels
        self.features = features

    def layers(self): 
        visual = []
        for i in range(len(self.network)):
            l = []
            for _ in range(len(self.network[i])):
                l.append('o')
            visual.append(l)
            
        print(*visual, sep='\n')


    def forward(self, inp):
        for i in self.network:
            for j in i:
                j.run(inp)
        calc = []
        for i in self.network[-1]:
            calc.append(1/(1+2.71828**-(i.value * i.weight + i.bias)))

        return sum(calc)

    def backward(self):
        error = []
        for i in range(len(self.labels)):
            prediction = self.network.forward(self.features[i])
            error.append((prediction-self.labels[i])**2)

        
    def bprop(self, prediction, label):
        error = 0.5*(prediction-label)**2
        der = prediction - label
        sigm = prediction(1-prediction)
            

test = Network(1, 2, 2)
test.layers()
print(test.forward(1))




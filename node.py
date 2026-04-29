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

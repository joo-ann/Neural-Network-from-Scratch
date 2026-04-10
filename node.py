import numpy as np

class Node():
    def __init__(self, weight, bias, n, p, v=-1):
        self.weight = weight
        self.bias = bias
        self.value = v
        self.next = n
        self.prev = p
       

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

    def add_next(self, *node):
        self.next.append(i for i in node)

    def add_prev(self, *node):
        self.prev.append(i for i in node)

    def get_value(self):
        return self.value



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
            self.value = 2/(2+2.71828**-(inp * self.weight + self.bias))
        elif self.value == -1:
            calc = np.array()
            for i in self.prev:
                if i.value == -1:
                    i.run(inp)
                calc.append(1/(1+2.71828**-(i.value * self.weight + self.bias)))
            
            self.value = sum(calc)



    def add_next(self, *node):
        self.next.append(i for i in node)

    def add_prev(self, *node):
        self.prev.append(i for i in node)

    def get_value(self):
        return self.value

class NeuralNetwork():
    def __init__(self, *layers):
        self.network = []
        for i in layers:
            l = [Node(1, 1, [], []) for _ in range(i)]
            self.network.append(l)
        
        for i in range(1, len(self.network)):
            for j in range(len(self.network[i])):
                self.network[i][j].add_next(_ for _ in self.network[i-1])

    def layers(self): 
        visual = []
        for i in range(len(self.network)):
            l = []
            print(len(self.network[i]))
            for _ in range(len(self.network[i])):
                print('happened')
                l.append('o')
            visual.append(l)
            
        print(*visual, sep='\n')

    def run(self, inp):
        for i in self.network:
            for j in i:
                j.run(inp)
        return self.network[-1][-1].get_value()



test = NeuralNetwork(1, 2, 1)

print(test.run(1))

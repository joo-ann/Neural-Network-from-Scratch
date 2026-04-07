import numpy as np
from node import Node


class Network():
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

    def forward(self, inp):
        for i in self.network:
            for j in i:
                j.run(inp)
        calc = []
        for i in self.network[-1]:
            calc.append(1/(1+2.71828**-(i.value * i.weight + i.bias)))

        return sum(calc)



test = Network(1, 2, 2)
print(test.forward(1))
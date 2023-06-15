import numpy as np
from settings import *

class Network:
    def __init__(self, *args):
        self.layers = Settings.layers
        self.activation = Settings.activation
        self.weights = []
        self.biases = []
        self.weighted_sums = []
        self.neurons = []

        if len(args) == 1:
            self.load_from_file(args[0])
        elif len(args) == 2:
            self.weights = args[0]
            self.biases = args[1]
        
    
    def load_from_file(self, f_weights: str):
        i = 0
        for line in open(f_weights, "r"):
            data = [float(x) for x in line.split(", ")]
            if (i % 2 == 0):
                # weights
                new_weight = []
                new_row = []
                for num in data:
                    new_row.append(num)
                    if len(new_row) >= Settings.layout[i//2]:
                        new_weight.append(new_row)
                        new_row = []

                new_weight = np.array(new_weight)

                # check if rows and columns match up
                if new_weight.shape != (Settings.layout[i//2+1], Settings.layout[i//2]):
                    print(f"the weights at line {i} do not match up with settings")
                    return
                    
                self.weights.append(new_weight)   
                
            else:
                # biases
                new_biases = []
                for num in data:
                    new_biases.append(num)
                new_biases = np.array(new_biases)

                # check if vector length match up
                if new_biases.shape != (Settings.layout[i//2+1],):
                    print(f"the biases at line {i} do not match up with settings")
                    return

                self.biases.append(new_biases)

            i += 1
    
    # serialize weights and biases into text file
    def save(self, outfile: str):
        file = open(outfile, "w")
        
        for i in range(0, self.layers - 1):
            weights = ", ".join(", ".join(str(y) for y in x) for x in self.weights[i])
            file.write(weights + "\n")
            biases = ", ".join(str(x) for x in self.biases[i])
            file.write(biases + "\n")
        
        file.close()
    
    # recursive feed
    def feed(self, input: np.ndarray) -> np.ndarray:
        self.weighted_sums = []
        self.neurons = []

        return self.forward_propagate(input, 0)
    
    def forward_propagate(self, input: np.ndarray, layer: int) -> np.ndarray:
        self.neurons.append(input)

        if layer >= self.layers - 1:
            return input
            
        # the sum of weighted activations plus bias
        weighted_sums = np.dot(self.weights[layer], input) + self.biases[layer][:, np.newaxis]
        # if second to last layer, then apply softmax instead of activation func
        if layer >= self.layers - 2:
            next_layer = softmax(weighted_sums) #only for use in multi-class classification problems
        elif self.activation == "relu":
            next_layer = np.vectorize(ReLU)(weighted_sums)
        elif self.activation == "sigmoid":
            next_layer = np.vectorize(sigmoid)(weighted_sums)
        elif self.activation == "tanh":
            next_layer = np.vectorize(tanh)(weighted_sums)
        
        self.weighted_sums.append(weighted_sums)
        return self.forward_propagate(next_layer, layer + 1)

            

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - tanh(x)**2

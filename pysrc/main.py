from Network import *
import numpy as np

net = Network("..\..\weights\weights.txt")

# feed to network
input = np.array([0,9999,9999,9999,0])
result = net.feed(input)
result = np.argmax(result)
print(result)
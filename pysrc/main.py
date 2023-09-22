from Network import *
import numpy as np

net = Network("..\weights\weights.txt")

# feed to network
input = np.array([4.8,3.1,1.6,0.2])
out_array = net.feed(input)
print(out_array)
result = []
for i in out_array:
    count = 0
    avg = 0
    for x in i:
        avg += x
        count += 1
    avg = avg/count
    i = avg
    result.append(i)
print(result.index(max(result)))
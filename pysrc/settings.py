# network layer settings

class Settings:
    train = True

    layers = 3
    layout = [5, 8, 3]
    activation = "relu" # relu, sigmoid, tanh, linear (linear not recommended for hidden layers)
    alpha = 0.5 # learning rate 0-1
    decay = 0.3 # decay rate 0-1
    batches = 200
    classification = True
    dropout_rate = .5 #.5 recommended, use 0 to bypass dropout

    if(classification):
            if(layout[-1] > 1):
                last_layer = "softmax"
            else:
                last_layer = "sigmoid"
    else:
        last_layer = "linear" #activation of final layer
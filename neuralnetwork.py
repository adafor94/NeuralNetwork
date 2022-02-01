from audioop import bias
import webbrowser
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, p = False):
        weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:], layer_sizes[:-1])]               #shapes of each weight matrix are neurons*inputs
        self.weights = [np.random.standard_normal(s)/s[1]**.5 for s in weight_shapes]           # s[1] = number of inputs to the layer. We want to balance the waits based on number of input    
        self.biases = [np.zeros((s,1)) for s in layer_sizes[1:]]                                # set all biases to zero to start with

        if p:
            print("Weights:")
            for w in self.weights:
                print(w, "\n")
            print("Biases:")
            for b in self.biases:
                print(b, "\n")      

    @staticmethod
    def activation(x):
        return 1/(1+np.exp(-x))         #sigmoid function 

    def predict(self, input):
        for w, b in zip(self.weights, self.biases):
            input = self.activation(np.matmul(w,input) + b)         # for each layer multiply weights of that layer and input to that layer. Add bias. Ax + b. 
        return input

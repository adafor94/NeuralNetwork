import numpy as np
import neuralnetwork as NN

with np.load('mnist.npz') as data:
    #print(data.files)
    training_images = data['training_images']
    training_labels = data['training_labels']

print(training_images.shape)

layer_sizes = (5,3,10)              # size of each layer
input = np.ones((layer_sizes[0],1))     # test input

my_network = NN.NeuralNetwork(layer_sizes, p = False)       # p=True if we want to print weights and biases.
print(my_network.predict(input))
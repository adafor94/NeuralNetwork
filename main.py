import numpy as np
import neuralnetwork as NN

with np.load('mnist.npz') as data:
    #print(data.files)
    training_images = data['training_images']
    training_labels = data['training_labels']

# print(training_images.shape)
# print(len(training_images[0]))
layer_sizes = (784,3,10)              # size of each layer
input = np.ones((layer_sizes[0],1))     # test input

my_network = NN.NeuralNetwork(layer_sizes, p = False)       # p=True if we want to print weights and biases.
# print("Predictions:")
# print(my_network.predict(training_images[0]))
# print("\nCorrect:")
# print(training_labels[0])

my_network.statistics(training_images, training_labels)

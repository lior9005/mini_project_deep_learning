import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
import part1.Utils as Utils
import Jack_test as jack_test

if __name__ == "__main__":
        # Load the data
        train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/GMMData.mat")

        ## Set the hyperparameters
        hidden_layer = [10, 10, 10]  # Set the hidden layer values manually
        batch_size=32
        epochs=200
        learning_rate=0.1

        jack_test.jac_test_layer(2, 3)
        jack_test.jac_test_resnet_layer(5)
        jack_test.jac_test_softmax_layer(2, 3)

        # layers = [train_data.shape[1]] + hidden_layer + [len(np.unique(train_labels))]
        # model = NeuralNetwork(layers, 'ReLU', True)
        # loss = model.train(train_data, train_labels, val_data, val_labels, batch_size, epochs, learning_rate)
        
        # plt.plot(loss)
        # plt.title("Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.grid(True)
        # plt.show()



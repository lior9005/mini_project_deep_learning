import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
import part1.Utils as Utils

if __name__ == "__main__":
        train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/GMMData.mat")
        layers = [train_data.shape[1], 10, len(np.unique(train_labels))]
        model = NeuralNetwork(layers, activation='ReLU')
        loss = model.train(train_data, train_labels, val_data, val_labels, batch_size=32, epochs=150, learning_rate=0.1)
        plt.plot(loss)
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()



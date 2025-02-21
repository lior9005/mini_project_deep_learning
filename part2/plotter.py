
from matplotlib import pyplot as plt
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import part1.Utils as Utils
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":

    def plot_accuracies(train_accuracies, val_accuracies, fileName):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(train_accuracies)), train_accuracies, label="Training Accuracy", linewidth=2)
        plt.plot(range(len(val_accuracies)), val_accuracies, label="Validation Accuracy", linewidth=2)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.title(f"{fileName} - accuracy", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.ylim(0, 1)
        plt.show()

    # # 2.4 GMM and Swiss Roll Data
    # train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/GMMData.mat")
    # hidden_layers = [train_data.shape[1]] + [50, 50, 50] + [len(np.unique(train_labels))]
    # model = NeuralNetwork(hidden_layers, 'ReLU', False)
    # loss, accuracy = model.train(train_data, train_labels, val_data, val_labels, 32, 200, 0.1)

    # plot_accuracies(loss, accuracy, "Datasets/GMMData.mat")

    # train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/PeaksData.mat")
    # hidden_layers = [train_data.shape[1]] + [10, 10, 10, 10, 10] + [len(np.unique(train_labels))]
    # model = NeuralNetwork(hidden_layers, 'ReLU', True)
    # loss, accuracy = model.train(train_data, train_labels, val_data, val_labels, 32, 200, 0.1)

    # plot_accuracies(loss, accuracy, "Datasets/PeaksData.mat")

    # # 2.5 Network lenghts with paramerter constraints
    # train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/GMMData.mat")
    # hidden_layers = [train_data.shape[1]] + [9, 9, 9] + [len(np.unique(train_labels))]
    # model = NeuralNetwork(hidden_layers, 'ReLU', True)
    # loss, accuracy = model.train(train_data, train_labels, val_data, val_labels, 32, 200, 0.1)

    # plot_accuracies(loss, accuracy, "Datasets/GMMData.mat")

    train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/PeaksData.mat")
    hidden_layers = [train_data.shape[1]] + [6, 6, 6] + [len(np.unique(train_labels))]
    model = NeuralNetwork(hidden_layers, 'ReLU', True)
    loss, accuracy = model.train(train_data, train_labels, val_data, val_labels, 32, 200, 0.1)

    plot_accuracies(loss, accuracy, "Datasets/PeaksData.mat")
import sys
import os
import time

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
import part1.Utils as Utils
import Jac_test
import matplotlib.pyplot as plt
import part1.Grad_test as grad_test


if __name__ == "__main__":
       
        train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/GMMData.mat")

        # 2.1 + 2.2
        Jac_test.jac_test_layer(2, 3)
        Jac_test.jac_test_resnet_layer(5)
        Jac_test.jac_test_softmax_layer(2, 3)
        
        # 2.3
        train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/PeaksData.mat")
        learning_rate = 0.1
        activation = 'TanH'
        resNet = False
        hidden_layer = [10, 10]
        model_layers = [train_data.shape[1]] + hidden_layer + [len(np.unique(train_labels))]

        model = NeuralNetwork(model_layers, activation, resNet)

        F = lambda X: model.calculate_loss(X, train_labels)
        g_F = lambda X: model.backward(model.forward(X), train_labels)

        grad_test.gradient_test_layer(F, g_F, train_data, "Gradient test for NN")

        # 2.4 Network lengths experiments
        data_sets = ["Datasets/GMMData.mat", "Datasets/SwissRollData.mat"]
        hidden_layers = [
                        [],
                        [10],
                        [10, 10, 10],
                        [10, 10, 10, 10, 10],
                        [50],
                        [50, 50, 50]
                        ]
        learning_rates = [0.1, 0.01, 0.001]
        batch_sizes = [32, 64, 128]
        epochs = 200

        for data_set in data_sets:
                train_data, train_labels, val_data, val_labels = Utils.load_data(data_set)
                for hidden_layer in hidden_layers:
                        layers = [train_data.shape[1]] + hidden_layer + [len(np.unique(train_labels))]
                        for learning_rate in learning_rates:
                                model = NeuralNetwork(layers, 'ReLU', False)
                                for batch_size in batch_sizes:
                                        start_time = time.time()
                                        loss, accuracy = model.train(train_data, train_labels, val_data, val_labels, batch_size, epochs, learning_rate)
                                        end_time = time.time()
                                        elapsed_time = end_time - start_time
                                        print(f"Data set: {data_set}, Hidden layers: {hidden_layer}, Learning rate: {learning_rate}, "
                                                f"Batch size: {batch_size}, Accuracy: {accuracy[-1]}, "
                                                f"Training time: {elapsed_time:.2f} seconds")

        # 2.5 Network lenghts with paramerter constraints
        epochs = 200
        activation = 'ReLU'
        batch_size = 32
        learning_rate = 0.1
        
        hidden_layers200 = [
                        [39],
                        [11, 11],
                        [5, 6, 7, 9],
                        [9, 7, 6, 5],
                        [4, 4, 4, 4, 4, 4, 4, 4, 4]
                        ]
        hidden_layers500 = [
                        [45],
                        [17, 17],
                        [5, 10, 12, 15],
                        [15, 12, 10, 5],
                        [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
                        ]
        hidden_layers200_resnet = [
                        [8, 8],
                        [6, 6, 6],
                        [3, 3, 3, 3, 3, 3, 3, 3, 3]
                        ]
        hidden_layers500_resnet = [
                        [13, 13],
                        [9, 9, 9],
                        [5, 5, 5, 5, 5, 5, 5, 5, 5]
                        ]

        train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/SwissRollData.mat")
        for hidden_layer in hidden_layers200:
                layers = [train_data.shape[1]] + hidden_layer + [len(np.unique(train_labels))]
                model = NeuralNetwork(layers, activation, False)
                loss, accuracy = model.train(train_data, train_labels, val_data, val_labels, batch_size, epochs, learning_rate)
                print(f"Data set: SwissRollData , resNet: False, layers: {[2] + hidden_layer + [2]}, Params: {Utils.calculate_total_params(layers, False)}, accuracy: {accuracy[-1]}")
        for hidden_layer in hidden_layers200_resnet:
                layers = [train_data.shape[1]] + hidden_layer + [len(np.unique(train_labels))]
                model = NeuralNetwork(layers, activation, True)
                loss, accuracy = model.train(train_data, train_labels, val_data, val_labels, batch_size, epochs, learning_rate)
                print(f"Data set: SwissRollData , resNet: True, layers: {[2] + hidden_layer + [2]}, Params: {Utils.calculate_total_params(layers, True)}, accuracy: {accuracy[-1]}")

        train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/GMMData.mat")
        for hidden_layer in hidden_layers500:
                layers = [train_data.shape[1]] + hidden_layer + [len(np.unique(train_labels))]
                model = NeuralNetwork(layers, activation, False)
                loss, accuracy = model.train(train_data, train_labels, val_data, val_labels, batch_size, epochs, learning_rate)
                print(f"Data set: GMMData , resNet: False, layers: {[5] + hidden_layer + [5]}, Params: {Utils.calculate_total_params(layers, False)}, accuracy: {accuracy[-1]}")
        for hidden_layer in hidden_layers500_resnet:
                layers = [train_data.shape[1]] + hidden_layer + [len(np.unique(train_labels))]
                model = NeuralNetwork(layers, activation, True)
                loss, accuracy = model.train(train_data, train_labels, val_data, val_labels, batch_size, epochs, learning_rate)
                print(f"Data set: GMMData , resNet: True, layers: {[5] + hidden_layer + [5]}, Params: {Utils.calculate_total_params(layers, True)}, accuracy: {accuracy[-1]}")
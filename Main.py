import numpy as np
import matplotlib.pyplot as plt
import NeuralNetwork
import scipy.io

# Load the .mat file

def main():
    # get input
    input_data = scipy.io.loadmat('your_file.mat')
    train_data = input_data['Yt']  # Training data (features)
    train_labels = input_data['Ct']  # Training labels
    test_data = input_data['Yv']  # Validation data (features)
    test_labels = input_data['Cv']  # Validation labels

    # Hyperparameters
    input_size = 2
    output_size = 3
    hidden_layers = [input_size, 16, 16, 16, output_size]
    learning_rate = 0.1
    epochs = 100
    batch_size = 10

    model_type = "res_net" #true if resnet or else

    model = NeuralNetwork(hidden_layers, model_type)

    # Train the neural network
    loss = model.train(train_data, train_labels, batch_size, learning_rate, epochs)

    #test
    acc = model.eval(test_data, test_labels)
    print("Accuracy: ", acc)

    sgd_demonstration (ex 2.1.2, 2.1.3)

    


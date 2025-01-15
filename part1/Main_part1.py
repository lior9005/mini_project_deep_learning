import numpy as np
import part1.Utils as Utils
import testFunc
import SGD 
import scipy.io

if __name__ == "__main__":

# classifier (1.1)
    # Generate data
    n_samples, n_features, n_classes = 100, 20, 5
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)  # Feature matrix
    y = np.random.randint(0, n_classes, size=n_samples)  # Labels
    W = np.random.randn(n_features, n_classes)  # Weights
    b = np.random.randn(n_classes)              # Biases

    # loss and gradient functions
    F = lambda W, b: Utils.softmax_loss(X, y, W, b)
    g_F = lambda W, b: Utils.softmax_gradient(X, y, W, b)

    print("Gradient Test for softmax loss")
    testFunc.gradient_test(F, g_F, W, b)
    print()

# SGD (1.2)
    print("synthetic SGD check:")
    samples, features = 100, 200
    SGD.run_synthetic_example(samples, features, lr=0.1, mini_batch_size=10, epochs=200)

# SGD (1.3)
    print("softmax SGD check:")
    train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/PeaksData.mat")
    #train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/SwissRollData.mat")
    #train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/GMMData.mat")
    train_acc, val_acc = SGD.softmax_sgd(train_data, train_labels, val_data, val_labels, lr=0.1, batch_size=32, epochs=100, lambda_=0.001)
    SGD.plot_accuracies(train_acc, val_acc)
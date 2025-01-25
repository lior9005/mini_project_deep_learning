import numpy as np
import Utils
import Grad_test
import SGD 
import os

if __name__ == "__main__":

# classifier (1.1)
    # Generate data
    n_samples, n_features, n_classes = 100, 20, 5
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)  # Feature matrix
    Y = np.random.randint(0, n_classes, size=n_samples)  # Labels
    W = np.random.randn(n_features, n_classes)  # Weights
    b = np.random.randn(1, n_classes)

    # loss and gradient functions
    F = lambda W, b: Utils.softmax_loss(X, Y, W, b)
    g_F = lambda W, b: Utils.softmax_gradient(X, Y, W, b)

    # print("Gradient Test for softmax loss")
    # Grad_test.softmax_gradient_test(F, g_F, W, b)
    # print()

# SGD (1.2)
    # print("synthetic SGD check:")
    # samples, features = 100, 200
    # SGD.run_synthetic_example(samples, features, lr, batch_size, epochs)

# SGD (1.3)
    print("SGD check:")
    lr= [0.001, 0.01, 0.1, 1]
    batch_size=[10, 50, 100, 200]
    epochs=300
    data_path = ["Datasets/GMMData.mat", "Datasets/PeaksData.mat", "Datasets/SwissRollData.mat"]
    for path in data_path:
        print(f"dataset: {os.path.basename(path)}")
        SGD.best_SGD_params(path, lr, batch_size, epochs)
        print()


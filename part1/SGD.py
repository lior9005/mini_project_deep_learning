import numpy as np
import matplotlib.pyplot as plt
import Utils as Utils

# setup the data
def setup_synthetic_data(m, n):
    # Generate random data matrix X of shape (m, n)
    X = np.random.randn(m, n)
    
    # Perform Singular Value Decomposition (SVD) on X
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Modify the singular values of X with random scaling
    S = np.exp(0.3 * np.random.randn(min(m, n)))
    
    # Reconstruct X using the modified singular values
    X = U @ np.diag(S) @ Vt
    
    # Generate the solution vector 'sol' and output 'y'
    sol = np.random.randn(n)
    y = X @ sol + 0.05 * np.random.randn(m)  # Add noise to the output
    
    # Regularization parameter
    lambda_ = 0.001
    
    # Solve the least squares problem with regularization
    I_n = np.eye(n)
    sol = np.linalg.solve((1.0 / m) * (X.T @ X) + lambda_ * I_n, (1.0 / m) * X.T @ y)
    
    return X, y, sol, lambda_

def synthetic_sgd(X, y, lambda_, lr, mini_batch_size, epochs):
    m, n = X.shape
    I_n = np.eye(n)
    
    # Initialize weights to zero
    w = np.zeros(n)
    
    # Batch size for mini-batch SGD
    mini_batch_size = 10
    loss = []

    # SGD loop
    for epoch in range(1, epochs):
        
        # Reduce the learning rate every 50 epochs
        if epoch % 50 == 0:
            lr *= 0.5
            print("Learning rate:", lr)

        #shuffle the data indices
        idxs = np.random.permutation(m)

        # Process mini-batches
        for k in range(m // mini_batch_size):
            Ib = idxs[k * mini_batch_size:(k + 1) * mini_batch_size]  # Mini-batch indices
            Xb = X[Ib, :]  # Mini-batch features
            grad = (1.0 / mini_batch_size) * Xb.T @ (Xb @ w - y[Ib]) + lambda_ * w  # Compute gradient
            w -= lr * grad  # Update weights
        
        # Compute the MSE for the entire dataset
        mse = Utils.compute_mse(w, X, y)
        
        # Store the MSE
        loss.append(mse)
    return loss

def sgd(X_train, y_train, X_val, y_val, lr, batch_size, epochs):
    num_features = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    #np.random.seed(42)  
    W = np.random.randn(num_features, num_classes)  # Initialize weights (n_features x n_classes)
    W /= np.linalg.norm(W)  # Normalize weights
    b = np.zeros((1, num_classes))
    
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # Shuffle data
        shuffled_indices = np.random.permutation(len(X_train))
        train_data = X_train[shuffled_indices]
        Y = y_train[shuffled_indices]

        # Mini-batch SGD
        for i in range(len(train_data) // batch_size):
            batch_X, batch_Y = get_batch(train_data, Y, batch_size, i)
            
            # Compute gradients for W and b
            dW, db = Utils.softmax_gradient(batch_X, batch_Y, W, b)

            # Update weights and biases
            W -= lr * dW
            b -= lr * db
        
        # Track training accuracy
        X_sample, Y_sample = Utils.get_samples(X_train, y_train, batch_size)
        train_acc = Utils.compute_accuracy(X_sample, Y_sample, W, b)
        train_accuracies.append(train_acc)
        
        # Track validation accuracy
        X_sample, Y_sample = Utils.get_samples(X_val, y_val, batch_size)
        val_acc = Utils.compute_accuracy(X_sample, Y_sample, W, b)
        val_accuracies.append(val_acc)

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Training Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}')
    return train_accuracies, val_accuracies

def plot_accuracies(train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_accuracies)), train_accuracies, label="Training Accuracy", linewidth=2)
    plt.plot(range(len(val_accuracies)), val_accuracies, label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Softmax SGD", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

# Function to run the SGD optimization and plot the performance
def run_synthetic_example(m, n, lr=0.1, mini_batch_size=10, epochs= 200):
    print(f"experimenting {m} samples with {n} features")
    
    # Setup synthetic data and problem
    X, y, sol, lambda_ = setup_synthetic_data(m, n)
    
    # Perform SGD and get the MSE at each iteration
    mse_sgd = synthetic_sgd(X, y, lambda_, lr, mini_batch_size, epochs)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mse_sgd)), mse_sgd, label='SGD', linewidth=2)
    
    # Adding plot labels and title
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('SGD', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

def get_batch(train_data, y, batch_size, batch_index):
    start = batch_index * batch_size
    end = start + batch_size
    return train_data[start:end], y[start:end]

if __name__ == "__main__":
    train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/GMMData.mat")
    #train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/SwissRollData.mat")
    #train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/GMMData.mat")
    train_acc, val_acc = sgd(train_data, train_labels, val_data, val_labels, lr=1, batch_size = 20, epochs=100)
    plot_accuracies(train_acc, val_acc)

import numpy as np
import matplotlib.pyplot as plt

# Function to setup the synthetic least squares problem
def setup(m, n):
    # Generate random data matrix X of shape (m, n)
    X = np.random.randn(m, n)
    
    # Perform Singular Value Decomposition (SVD) on X
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Modify the singular values of X with random scaling
    S = np.exp(0.3 * np.random.randn(min(m, n)))
    
    # Reconstruct X using the modified singular values
    X = U @ np.diag(S) @ Vt
    print("Cond:", np.max(S) / np.min(S))  # Print condition number of X
    
    # Generate the solution vector 'sol' and output 'y'
    sol = np.random.randn(n)
    y = X @ sol + 0.05 * np.random.randn(m)  # Add noise to the output
    
    # Regularization parameter
    lambda_ = 0.001
    
    # Solve the least squares problem with regularization
    I_n = np.eye(n)
    sol = np.linalg.solve((1.0 / m) * (X.T @ X) + lambda_ * I_n, (1.0 / m) * X.T @ y)
    
    return X, y, sol, lambda_

# Function to perform SGD and plot the performance
def sgd(X, y, lambda_):
    m, n = X.shape
    I_n = np.eye(n)
    
    # Initialize weights to zero
    w = np.zeros(n)
    
    # Batch size for mini-batch SGD
    batch = 10
    
    # List to store the norm of the gradient at each step
    norms = [np.linalg.norm(y)]
    
    # Learning rate
    alpha = 10.0
    lr = alpha
    
    # Maximum number of iterations
    max_epochs = 500
    
    # SGD loop
    for epoch in range(1, max_epochs + 1):
        # Shuffle the data indices
        idxs = np.random.permutation(m)
        
        # Reduce the learning rate every 50 epochs
        if epoch % 50 == 0:
            lr *= 0.5
            print("Learning rate:", lr)
        
        # Process mini-batches
        for k in range(m // batch):
            Ib = idxs[k * batch:(k + 1) * batch]  # Mini-batch indices
            Xb = X[Ib, :]  # Mini-batch features
            grad = (1.0 / batch) * Xb.T @ (Xb @ w - y[Ib]) + lambda_ * w  # Compute gradient
            w -= lr * grad  # Update weights
        
        # Compute the norm of the gradient for the entire dataset
        nn = np.linalg.norm((1 / m) * X.T @ (X @ w - y) + lambda_ * w)
        
        # Store the gradient norm
        norms.append(nn)
    
    return norms

# Function to run the SGD optimization and plot the performance
def run_and_plot(m, n):
    print(f"Running setup for m={m}, n={n}")
    
    # Setup synthetic data and problem
    X, y, sol, lambda_ = setup(m, n)
    
    # Perform SGD and get the norms at each iteration
    norms_sgd = sgd(X, y, lambda_)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(norms_sgd)), norms_sgd, label='SGD', linewidth=2)
    
    # Adding plot labels and title
    plt.xlabel('Iterations/Epochs', fontsize=14)
    plt.ylabel('Norm of Error', fontsize=14)
    plt.title('SGD Performance', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

# Example usage with different dataset sizes
n = 200  # Number of features
for m in [100, 400]:  # Varying number of samples
    run_and_plot(m, n)

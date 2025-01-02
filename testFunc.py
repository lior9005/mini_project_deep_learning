import numpy as np
import matplotlib.pyplot as plt
import Funcs

# Gradient test function
def gradient_test(F, g_F, W, d, title, epsilon=0.1, max_iter=8):
    F0 = F(W)               # Initial loss value
    g0 = g_F(W)             # Analytical gradient at W
    y0 = []                 # Errors for zero-order approximation
    y1 = []                 # Errors for first-order approximation

    print("k\t\terror order 1 \t\t error order 2")
    for k in range(max_iter):
        epsk = epsilon * (0.5 ** k)  # Decreasing step sizes
        Fk = F(W + epsk * d)         # Perturbed loss value
        F1 = F0 + epsk * np.sum(g0 * d)  # Linear approximation

        y0.append(abs(Fk - F0))      # Zero-order error
        y1.append(abs(Fk - F1))      # First-order error
        print(f"{k}\t\t{y0[-1]:.6e}\t\t{y1[-1]:.6e}")

    # Plotting the errors
    plt.figure()
    plt.semilogy(range(max_iter), y0, label="Zero order approx (O(ϵ))")
    plt.semilogy(range(max_iter), y1, label="First order approx (O(ϵ²))")
    plt.legend()
    plt.title(f"Gradient Test: {title}")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.grid()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    n_samples, n_features, n_classes = 100, 20, 5
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)  # Feature matrix
    y = np.random.randint(0, n_classes, size=n_samples)  # Labels
    W = np.random.randn(n_features, n_classes)  # Weights
    b = np.random.randn(n_classes)              # Biases

    # Define loss and gradient functions for testing W
    F_W = lambda W: Funcs.softmax_loss(X, y, W, b)
    g_F_W = lambda W: Funcs.softmax_gradient(X, y, W, b)[0]  # Gradient w.r.t. W

    # Random direction for perturbation
    d_W = np.random.randn(*W.shape)

    print("Gradient Test for W:")
    gradient_test(F_W, g_F_W, W, d_W, "weights")

    # Define loss and gradient functions for testing b
    F_b = lambda b: Funcs.softmax_loss(X, y, W, b)
    g_F_b = lambda b: Funcs.softmax_gradient(X, y, W, b)[1]  # Gradient w.r.t. b

    # Random direction for perturbation
    d_b = np.random.randn(*b.shape)

    print("\nGradient Test for b:")
    gradient_test(F_b, g_F_b, b, d_b, "biases")

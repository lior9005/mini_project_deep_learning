import numpy as np
import part1.Utils as Utils
import NeuralNetwork
import part1.Grad_test as grad_test

def jac_test_layer(in_dim, out_dim):
    X_rand = np.random.randn(1, in_dim)
    u = np.random.randn(out_dim)
    W, W2, b = initialize_weight_and_bias(in_dim, out_dim)

    def g_x(X):
        X_next = np.dot(X, W) + b
        X_next = np.tanh(X_next)
        g_X_u = np.dot(X_next, u)
        return g_X_u
    
    def gradient_g_x(X):
        X_next = np.dot(X, W) + b
        sigma_prime = 1 - np.tanh(X_next) ** 2
        sigma_prime_u = sigma_prime * u
        grad_X = np.dot(sigma_prime_u, W.T)
        return grad_X

    grad_test.gradient_test_layer(g_x, gradient_g_x, X_rand, 'Jacobian Gradient Test')

def jac_test_resnet_layer(dim):
    X_rand = np.random.randn(1, dim)
    u = np.random.randn(dim)
    W, W2, b = initialize_weight_and_bias(dim, dim)

    def g_x(X):
        X_next = np.dot(X, W) + b
        X_next = np.tanh(X_next)
        X_next = X + np.dot(X_next, W2)
        g_X_u = np.dot(X_next, u)
        return g_X_u
    
    def gradient_g_x(X):
        X_next = np.dot(X, W) + b
        sigma_prime = 1 - np.tanh(X_next) ** 2
        sigma_prime_W2T_u = sigma_prime * np.dot(u, W2.T)
        grad_X = u + np.dot(sigma_prime_W2T_u, W.T)
        return grad_X

    grad_test.gradient_test_layer(g_x, gradient_g_x, X_rand, 'Jacobian Gradient Test - ResNet')

def jac_test_softmax_layer(in_dim, out_dim):
    X_rand = np.random.randn(1, in_dim)
    Y = np.random.randint(0, out_dim, size=1)
    W, W2, b = initialize_weight_and_bias(in_dim, out_dim)

    def g_x(X):
        X_next = np.dot(X, W) + b
        softmax_X = np.exp(X_next - np.max(X_next, axis=1, keepdims=True))
        softmax_X /= np.sum(softmax_X, axis=1, keepdims=True)
        pred_probs = softmax_X[np.arange(Y.shape[0]), Y]
        loss = -(np.log(pred_probs))
        return loss
    
    def gradient_g_x(X):
        X_next = np.dot(X, W) + b
        softmax_X = np.exp(X_next - np.max(X_next, axis=1, keepdims=True))
        softmax_X /= np.sum(softmax_X, axis=1, keepdims=True)

        softmax_X[0, Y] -= 1 #substract 1 from the correct class probabilty for each input
        grad_X = np.dot(softmax_X, W.T)

        return grad_X

    grad_test.gradient_test_layer(g_x, gradient_g_x, X_rand, 'Jacobian Gradient Test - Softmax')

def initialize_weight_and_bias(in_dim, out_dim):
    W = np.random.randn(in_dim, out_dim)
    W /= np.linalg.norm(W)  # Normalize weights
    W2 = np.random.randn(in_dim, out_dim)
    W2 /= np.linalg.norm(W2)
    b = np.random.randn(1, out_dim)
    b /= np.linalg.norm(b)
    return W, W2, b



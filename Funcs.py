import numpy as np

def ReLU(x):
    return np.maximum(0, x)

def TanH(x):
    return np.tanh(x)

def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

def TanH_derivative(x):
    return 1 - np.tanh(x) ** 2

# Softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Loss function for softmax regression
def softmax_loss(X, y, W, b):
    m = X.shape[0]  # Number of samples
    z = X.dot(W) + b
    probs = softmax(z)
    correct_log_probs = -np.log(probs[range(m), y])
    loss = np.sum(correct_log_probs) / m
    return loss


# Gradient of the loss function with respect to W and b
def softmax_gradient(X, y, W, b):
    m = X.shape[0]
    z = X.dot(W) + b
    probs = softmax(z)
    probs[range(m), y] -= 1
    dW = X.T.dot(probs) / m
    db = np.sum(probs, axis=0) / m
    return dW, db
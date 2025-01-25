import numpy as np
import scipy.io

def compute_mse(w, X, y):
    errors = X @ w - y
    return np.mean(errors**2)

def compute_accuracy_part1(X, y, W, b):
    logits = X.dot(W) + b  # Compute logits
    probs = softmax(logits)  # Compute probabilities
    predictions = np.argmax(probs, axis=1)  # Get class predictions
    return np.mean(predictions == y)  # Compare with true labels

def load_data(path):
    dataset = scipy.io.loadmat(path)
    train_data = dataset['Yt'].T  
    train_labels = np.argmax(dataset['Ct'], axis=0).astype(int)
    val_data = dataset['Yv'].T
    val_labels = np.argmax(dataset['Cv'], axis=0).astype(int)
    return train_data, train_labels, val_data, val_labels

def softmax(X):
    exp_x = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

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
    z = np.dot(X , W) + b
    probs = softmax(z)
    probs[np.arange(m), y] -= 1
    dW = X.T.dot(probs) / m
    db = np.sum(probs, axis=0) / m
    return dW, db

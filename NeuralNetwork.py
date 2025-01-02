import numpy as np

class NeuralNetwork:
    def _init_(self, layers, model_type):
        self.layers = layers
        self.X_arrays = []
        self.gradient_B = []
        self.gradient_W = []
        self.model_type = model_type
        self.weights, self.biases = self.initialize_weights_and_biases()

    def softmax(X):
        exp_x = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.X_arrays = [X]
        for w, b in zip(self.weights, self.biases):
            X = self.ReLU(np.dot(X, w) + b)
            self.X_arrays.append(X)
        return self.softmax(X)

    # do we need the derivatives of the loss for w and b? maybe only for x is enough
    # check the range of the for loop, maybe there is too many iterations or indexes not correct
    def backward(self, X_softmax, Y, learning_rate):
        self.gradient_W = []
        self.gradient_B = []

        v = self.derivative_loss_X(X_softmax, Y)
        dW = self.derivative_loss_W(X_softmax, Y)
        dB = self.derivative_loss_B(X_softmax, Y)
        self.gradient_W.insert(0, dW)
        self.gradient_B.insert(0, dB)

        for i in range(len(self.layers) - 2, -1, -1):
            dW = self.derivative_W(v, i)
            dB = self.derivative_B(v, i)
            self.gradient_W.insert(0, dW)
            self.gradient_B.insert(0, dB)

            v = self.derivative_X(v, i)

        self.update_weights_biases(learning_rate)

    # fix the way to get batch
    def train(self, train_data, y, batch_size, epochs, learning_rate):
        loss_list = []
        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(len(train_data))
            train_data = train_data[shuffled_indices]
            y = y[shuffled_indices]
            for i in range(len(train_data) // batch_size):
                train_X, train_Y = self.get_batch(train_data, y, batch_size, i)
                X_soft = self.forward(train_X)
                loss = self.calculate_loss(X_soft, train_Y)
                loss_list.append(loss)
                self.backward(X_soft, train_Y, learning_rate)
        return loss_list

    def eval(self, test_data):
        return self.forward(test_data)

    def derivative_X(self, v, index):
        W = self.weights[index]
        B = self.biases[index]
        X = self.X_arrays[index]
        sigma_prime = self.ReLU_derivative(np.dot(W, X) + B)
        return np.dot(W.T, sigma_prime * v)

    def derivative_B(self, v, index):
        W = self.weights[index]
        B = self.biases[index]
        X = self.X_arrays[index]
        sigma_prime = self.ReLU_derivative(np.dot(X, W) + B)
        return sigma_prime * v

    def derivative_W(self, v, index):
        W = self.weights[index]
        B = self.biases[index]
        X = self.X_arrays[index]
        sigma_prime = self.ReLU_derivative(np.dot(X, W) + B)
        return np.dot(sigma_prime * v, X.T)

    def derivative_loss_X(self, X_soft, Y):
        m = X_soft.shape[0]
        W = self.weights[self.weights.size() - 1]
        soft_minus_C = X_soft - Y
        return (1 / m) * np.dot(W, soft_minus_C.T)

    def derivative_loss_W(self, X_soft, Y):
        m = X_soft.shape[0]
        X = self.weights[self.X_arrays.size() - 1]
        soft_minus_C = X_soft - Y
        return (1 / m) * np.dot(X, soft_minus_C)

    def derivative_loss_B(self, X_soft, Y):
        # implement

    def calculate_loss(self, X_soft, Y):
        X_soft_log = np.log(X_soft)
        return Y * X_soft_log

    def initialize_weights_and_biases(self):
        weights = []
        biases = []
        for i in range(len(self.layers) - 1):
            n_1 = self.layers[i+1]
            n_2 = self.layers[i]
            W = np.random.randn(n_1, n_2)
            current_norm = np.linalg.norm(W)
            W /= current_norm
            weights.append(W)
            biases.append(np.zeros((n_1, 1)))
        return weights, biases

    def update_weights_biases(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.gradient_W[i]
            self.biases[i] -= learning_rate * self.gradient_B[i]

    def get_batch(self, train_data, y, batch_size, batch_index):
        start = batch_index * batch_size
        end = start + batch_size
        return train_data[start:end], y[start:end]
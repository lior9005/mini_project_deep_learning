import numpy as np

class NeuralNetwork:
    def __init__(self, layers, activation, is_resNet = False):
        self.layers = layers
        self.X_arrays = []
        self.gradient_B = []
        self.gradient_W = []
        self.gradient_W2 = []
        self.is_resNet = is_resNet
        self.weights, self.weights2, self.biases = self.initialize_weights_and_biases()

        activation_functions = {
            "ReLU": self.ReLU,
            "TanH": self.TanH
        }
        self.activation = activation_functions[activation]
        self.check_resNet()

    def check_resNet(self):
        if self.is_resNet:
            if len(self.layers) < 4:
                raise ValueError("For ResNet, the number of hidden layers must be 2 or more.")
            hidden_layer_size = self.layers[1]
            for layer_size in self.layers[1:-1]:
                if layer_size != hidden_layer_size:
                    raise ValueError("For ResNet, all hidden layers must have the same size.")
                
    def softmax(self, X):
        exp_x = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.X_arrays = [X]
        for i in range(len(self.layers) - 2):
            W = self.weights[i]
            b = self.biases[i]
            if i==0 or self.is_resNet == False:
                X = self.activation(np.dot(X, W) + b, False)
            else:
                W2 = self.weights2[i]
                X = X + np.dot(self.activation(np.dot(X, W) + b, False), W2)
            self.X_arrays.append(X)
        W = self.weights[-1]
        b = self.biases[-1]
        X_soft = self.softmax(np.dot(X, W) + b)
        return X_soft

    # check the range of the for loop, maybe there is too many iterations or indexes not correct
    def backward(self, X_softmax, Y, learning_rate):
        self.gradient_W = []
        self.gradient_W2 = []
        self.gradient_B = []

        v = self.softmax_gradients(X_softmax, Y)

        if self.is_resNet:
            for i in range(len(self.layers) - 3, 0, -1):
                v = self.resNet_layer_gradients(v, i)
            v = self.layer_gradients(v, 0)
            ## dummy gradient for the first layer
            self.gradient_W2.insert(0, self.weights2[0])
        else:
            for i in range(len(self.layers) - 3, -1, -1):
                v = self.layer_gradients(v, i)

        self.update_weights_biases(learning_rate)

    # fix the way to get batch
    def train(self, train_data, Y, X_val, Y_val, batch_size, epochs, learning_rate):
        loss_list = []
        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(len(train_data))
            train_data = train_data[shuffled_indices]
            Y = Y[shuffled_indices]
            for i in range(len(train_data) // batch_size):
                train_X, train_Y = self.get_batch(train_data, Y, batch_size, i)
                X_soft = self.forward(train_X)
                self.backward(X_soft, train_Y, learning_rate)

            ## after finishing each epoch, take a random batch and calculate the loss
            loss = self.calculate_loss(X_val, Y_val)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}, loss: {loss:.4f}')
            loss_list.append(loss)
                
        return loss_list

    def eval(self, test_data):
        return self.forward(test_data)


    def layer_gradients(self, v, index):
        W = self.weights[index]
        b = self.biases[index]
        X = self.X_arrays[index]
        m = X.shape[0]
        sigma_prime = self.activation(np.dot(X, W) + b, True)
        sigma_prime_v = sigma_prime * v

        v = np.dot(sigma_prime_v, W.T)
        ## do we need to divide by m or not?
        dW = np.dot(X.T, sigma_prime_v) / m
        ## do we need to divide by m or not?
        db = np.sum(sigma_prime_v, axis=0, keepdims=True) / m
        
        self.gradient_W.insert(0, dW)
        self.gradient_B.insert(0, db)
        
        return v
    
    def resNet_layer_gradients(self, v, index):
        W = self.weights[index]
        W2 = self.weights2[index]
        b = self.biases[index]
        X = self.X_arrays[index]
        X_next = self.X_arrays[index + 1]
        m = X.shape[0]
        sigma_prime = self.activation(np.dot(X, W) + b, True)
        sigma_prime_W2T_v = sigma_prime * np.dot(v, W2.T)

        ## do we need to divide by m or not?
        dW = np.dot(X.T, sigma_prime_W2T_v) / m
        dW2 = np.dot(X_next.T, v) / m
        ## do we need to divide by m or not?
        db = np.sum(sigma_prime_W2T_v, axis=0, keepdims=True) / m
        v = v + np.dot(sigma_prime_W2T_v, W.T)

        self.gradient_W.insert(0, dW)
        self.gradient_W2.insert(0, dW2)
        self.gradient_B.insert(0, db)
        
        return v

    def softmax_gradients(self, X_soft, Y):
        m = X_soft.shape[0]
        W = self.weights[-1]
        X = self.X_arrays[-1]
        soft_minus_C = X_soft
        soft_minus_C[np.arange(m), Y] -= 1 #substract 1 from the correct class probabilty for each input
        soft_minus_C /= m
        v = np.dot(soft_minus_C, W.T)
        dW = np.dot(X.T, soft_minus_C)
        db = np.sum(soft_minus_C, axis=0, keepdims=True)

        self.gradient_W.insert(0, dW)
        self.gradient_B.insert(0, db)

        if self.is_resNet:
            self.gradient_W2.insert(0, dW)   ## dummy gradient for the last layer

        return v

    def calculate_loss(self, X, Y):
        X_soft = self.forward(X)
        pred_probs = X_soft[np.arange(Y.shape[0]), Y]
        loss = -np.mean(np.log(pred_probs))
        return loss
    
    def calculate_accuracy(self, X, Y):
        X_soft = self.forward(X)
        class_predictions = np.argmax(X_soft, axis=1) # get the predicted class for each input
        correct = np.sum(class_predictions == Y)
        accuracy = correct / Y.shape[0]
        return accuracy

    def initialize_weights_and_biases(self):
        weights = []
        weights2 = []
        biases = []
        for i in range(len(self.layers) - 1):
            n_1 = self.layers[i+1]
            n_2 = self.layers[i]
            W = np.random.randn(n_2, n_1)
            W /= np.linalg.norm(W)  # Normalize weights
            W2 = np.random.randn(n_2, n_1)
            W2 /= np.linalg.norm(W2)
            weights.append(W)
            weights2.append(W2)
            biases.append(np.zeros((1, n_1)))
        return weights, weights2, biases

    def update_weights_biases(self, learning_rate):
        for i in range(len(self.layers) - 1):
            self.weights[i] -= learning_rate * self.gradient_W[i]
            self.biases[i] -= learning_rate * self.gradient_B[i]
            if self.is_resNet:
                self.weights2[i] -= learning_rate * self.gradient_W2[i]

    def get_batch(self, train_data, y, batch_size, batch_index):
        start = batch_index * batch_size
        end = start + batch_size
        return train_data[start:end], y[start:end]
    
    def ReLU(self, X, derivative):
        if derivative:
            return np.where(X > 0, 1, 0)
        return np.maximum(0, X)

    def TanH(self, x, derivative):
        if derivative:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)
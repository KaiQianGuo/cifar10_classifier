# 文件结构：three_layer_net.py
# 功能：三层神经网络结构（前向 + 反向传播）

import numpy as np

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', reg_lambda=0.0):
        self.params = {}
        self.params['W1'] = 0.01 * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros((1, hidden_size))
        self.params['W2'] = 0.01 * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros((1, output_size))
        self.activation = activation
        self.reg_lambda = reg_lambda

    def _activation(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        else:
            raise ValueError("Unsupported activation function")

    def _activation_grad(self, z):
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)

    def softmax(self, z):
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        z1 = X.dot(W1) + b1
        a1 = self._activation(z1)
        z2 = a1.dot(W2) + b2
        a2 = self.softmax(z2)

        self.cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        return a2

    def compute_loss_and_accuracy(self, X, y):
        num_samples = X.shape[0]
        probs = self.forward(X)
        correct_logprobs = -np.log(probs[range(num_samples), y] + 1e-8)
        data_loss = np.sum(correct_logprobs) / num_samples

        # L2 regularization
        reg_loss = 0.5 * self.reg_lambda * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        loss = data_loss + reg_loss

        y_pred = np.argmax(probs, axis=1)
        acc = np.mean(y_pred == y)
        return loss, acc

    def backward(self, y):
        grads = {}
        X, z1, a1, z2, a2 = self.cache['X'], self.cache['z1'], self.cache['a1'], self.cache['z2'], self.cache['a2']
        
        num_samples = X.shape[0]
        delta2 = a2
        delta2[range(num_samples), y] -= 1
        delta2 /= num_samples

        grads['W2'] = a1.T.dot(delta2) + self.reg_lambda * self.params['W2']
        grads['b2'] = np.sum(delta2, axis=0, keepdims=True)

        delta1 = delta2.dot(self.params['W2'].T) * self._activation_grad(z1)
        grads['W1'] = X.T.dot(delta1) + self.reg_lambda * self.params['W1']
        grads['b1'] = np.sum(delta1, axis=0, keepdims=True)

        return grads

    def update(self, grads, learning_rate):
        for param in self.params:
            self.params[param] -= learning_rate * grads[param]

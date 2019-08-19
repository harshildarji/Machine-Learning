# Neural Network from scratch
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return (1.0 / (1 + np.exp(-x)))

def sigmoid_derivative(d):
    return (d * (1 - d))

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.w1 = np.random.rand(self.input.shape[1], 4)
        self.w2 = np.random.rand(4, 1)
        self.y = y
        self.op = np.zeros(y.shape)
        self.loss = []

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.w1))
        self.op = sigmoid(np.dot(self.layer1, self.w2))
        self.loss.append(np.sum((self.y - self.op) ** 2))

    def backprop(self):
        d_w2 = np.dot(self.layer1.T, (2 * (self.y - self.op) * sigmoid_derivative(self.op)))
        d_w1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.op) * sigmoid_derivative(self.op), self.w2.T) * sigmoid_derivative(self.layer1))) 
        
        self.w1 += d_w1
        self.w2 += d_w2

if __name__ == '__main__':
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(X, y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(nn.op)
    
    plt.plot(range(0, 1500), nn.loss)
    plt.grid(True)
    plt.show()
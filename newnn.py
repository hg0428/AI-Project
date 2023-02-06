import numpy
import random
import json

# the sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))

# the derivative of the sigmoid function
def sigmoidPrime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# sigmoid vectors
sigmoidVector = numpy.vectorize(sigmoid)
sigmoidPrimeVector = numpy.vectorize(sigmoidPrime)

#A class that implements stochastic gradient descent learning algorithm for a feedforward neural network
class NN:
    def __init__(self, sizes):
        self.numLayers = len(sizes)
        self.sizes = sizes

        # the biases and weights for the network are initialized randomly, using a Guassian distribution with mean 0, and variance 1
        self.biases = [numpy.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        print(self.biases, self.weights, self.numLayers)

    # feedForward function - return the output of the network
    def feedForward(self, inputs):
        for b, w in zip(self.biases, self.weights):
            inputs = sigmoidVector(numpy.dot(w, inputs) + b)
        return inputs

    # train function - train the neural network using mini-batch stochastic gradient descent
    # the trainingData is a list of tuples "(x, y)" representing the training inputs and the desired outputs
    # if testData is provided then the network will be evaluated against the test data after each epoch
    def train(self, trainingData, epochs, miniBatchSize, eta, testData = None):
        if testData:
            nTest = len(testData)

        n = len(testData)
        for j in xrange(epochs):
            random.shuffle(trainingData)

            miniBatches = [trainingData[k:k + miniBatchSize] for k in xrange(0, n, miniBatchSize)]
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta)

            if testData:
                print("Epoch %i: %i / %i" % (j, self.evaluate(testData), nTest))
            else:
                print("Epoch %i complete" % j)

                
    # updateMiniBatch function - update the network's weights and biases by applying gradient descent using backpropagation
    # to a single mini batch
    # the miniBatch is a list of tuples "(x, y)" and eta is the learning rate
    def updateMiniBatch(self, miniBatch, eta):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        for x, y in miniBatch:
            delta_nabla_b, delta_nabla_w = self.backPropagate(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(miniBatch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(miniBatch)) * nb for b, nb in zip(self.biases, nabla_b)]

    # backPropagate function - returns a tuple "(nabla_b, nabla_w)" representing the gradient for the cost function C_x
    # nabla_b and nabla_w are layer-by-layer lists of numpy arrays, similar to self.biases and self.weights
    def backPropagate(self, x, y):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        # feedForward
        activation = x
        activations = [x] # list to store all of the activations, layer by layer
        zs = [] # list to store all of the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) + b
            zs.append(z)

            activation = sigmoidVector(z)
            activations.append(activation)

        # backward pass
        delta = self.costDerivative(activations[-1], y) * sigmoidPrimeVector(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.numLayers):
            z = zs[-l]
            spv = sigmoidPrimeVector(z)

            delta = numpy.dot(self.weights[-l + 1].transpose(), delta) * spv

            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    # evaluate function - return the number of test inputs for which the neural network outputs the correct result
    def evaluate(self, testData):
        testResults = [(numpy.argmax(self.feedForward(x)), y) for (x, y) in testData]
        return sum(int(x == y) for (x, y) in testResults)

    # costDerivative function - return the vector of partial derivatives for the output activations
    def costDerivative(self, outputActivations, y):
        return (outputActivations - y)

    # save function - save the neural network to filename
    def save(self, filename):
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }

        with open(filename, "w") as handle:
            json.dump(data, handle)

# load function - load a neural network from the file filename
# returns a network instance
def load(filename):
    with open(filename, "r") as handle:
        data = json.load(handle)

    network = NN(data["sizes"])
    network.weights = [numpy.array(w) for w in data["weights"]]
    network.biases = [numpy.array(b) for b in data["biases"]]

    return network


if __name__ == "__main__":
    x = NN((3, 3, 3, 3, 3, 3))
    x.train()
    print(x.feedForward([0, 1, 1]))
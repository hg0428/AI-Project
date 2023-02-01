import numpy as np
import pickle


class NeuralNetwork:
    def __init__(self, dim=4, synaptic_weights=None):
        if synaptic_weights is None:
            # seeding for random number generation
            np.random.seed(1)

            # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
            self.synaptic_weights = 2 * np.random.random((dim, 1)) - 1
        else:
            self.synaptic_weights = np.array(synaptic_weights)

    def sigmoid(self, x):
        # applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # computing derivative to the Sigmoid function
        return x * (1 - x)

    def adjust(self, training_inputs, training_outputs):
        # siphon the training data via  the neuron
        output = self.think(training_inputs)

        # computing error rate for back-propagation
        error = training_outputs - output

        # performing weight adjustments
        adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

        self.synaptic_weights += adjustments

    def train(self, training_inputs, training_outputs, training_iterations):
        training_inputs = np.array(training_inputs)
        training_outputs = np.array([training_outputs]).T
        # training the model to make accurate predictions while adjusting weights continually
        for _ in range(training_iterations):
            self.adjust(training_inputs, training_outputs)

    def think(self, inputs):
        # passing the inputs via the neuron to get output
        # converting values to floats
        inputs = np.array(inputs).astype(float)
        return self.sigmoid(np.dot(inputs, self.synaptic_weights))

    def addInput(self, start=0):
        if start is None:
            start = sum(x[0] for x in self.synaptic_weights.tolist()) / len(
                self.synaptic_weights
            )
        self.synaptic_weights = np.array([[start]] + self.synaptic_weights.tolist())

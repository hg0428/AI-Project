from numpy import random, array, exp, dot
import pickle


def sigmoid(x):
        # applying the sigmoid function
        return 1 / (1 + exp(-x))

def sigmoid_derivative(self, x):
        # computing derivative to the Sigmoid function
        return x * (1 - x)
        
class NeuralNetwork:
    def __init__(self, dim=4, synaptic_weights=None):
        if synaptic_weights == None:
            # seeding for random number generation
            random.seed(1)

            # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
            self.synaptic_weights = 2 * random.random((dim, 1)) - 1
        else:
            self.synaptic_weights = array(synaptic_weights)

    def adjust(self, training_inputs, training_outputs):
        # siphon the training data via  the neuron
        output = self.think(training_inputs)

        # computing error rate for back-propagation
        error = training_outputs - output

        # performing weight adjustments
        adjustments = dot(training_inputs.T, error * sigmoid_derivative(output))

        self.synaptic_weights += adjustments
        return output

    def train(self, training_inputs, training_outputs, training_iterations):
        training_inputs = array(training_inputs)
        training_outputs = array([training_outputs]).T
        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            self.adjust(training_inputs, training_outputs)

    def think(self, inputs):
        # passing the inputs via the neuron to get output
        # converting values to floats
        inputs = array(inputs).astype(float)
        output = sigmoid(dot(inputs, self.synaptic_weights))
        return output

    def addInput(self, start=0):
        if start == None:
            start = sum([x[0] for x in self.synaptic_weights.tolist()]) / len(
                self.synaptic_weights
            )
        self.synaptic_weights = array([[start]] + self.synaptic_weights.tolist())

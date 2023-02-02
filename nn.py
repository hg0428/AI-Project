from numpy import random, array, exp, dot, ndarray
import pickle


def sigmoid(x):
    # applying the sigmoid function
    return 1 / (1 + exp(-x))


def sigmoid_derivative(x):
    # computing derivative to the Sigmoid function
    return x * (1 - x)


def fill(l, length, null=0, reverse=False):
    if len(l) > length:
        if reverse:
            return l[len(l) - length :]
        else:
            return l[:length]
    else:
        for x in range(length - len(l)):
            if reverse:
                l = [null] + l
            else:
                if isinstance(l, str):
                    l += str(null)
                else:
                    l.append(null)
    return l


def process_value(x, bpc=8):
    if type(x) == str:
        return [int(i) for i in "".join([format(ord(i), f"0{bpc}b") for i in x])]
    elif type(x) == int:
        return [int(i) for i in format(x, "b")]
    elif type(x) == list:
        return x
    elif type(x) == array:
        return x
    elif type(x) == ndarray:
        return x


def decode(data, bpc=8):
    out = [round(x) for x in data]
    bytes = [out[x : x + bpc] for x in range(0, len(out), bpc)]
    strbytes = ["".join([str(i) for i in x]) for x in bytes]
    chrs = [int(x, 2) for x in strbytes]
    string = ""
    for x in chrs:
        if x == 0:
            string += ""
        try:
            string += chr(x)
        except:
            string += ""
    return string


class NeuralNetwork:
    def __init__(self, dim=4, outs=1, bpc=8, synaptic_weights=None):
        self.max_input_length = self.dim = dim
        self.max_output_length = self.outs = outs
        self.bpc = bpc
        if synaptic_weights == None:
            # seeding for random number generation
            random.seed(1)

            # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
            self.synaptic_weights = 2 * random.random((dim, outs)) - 1
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
        training_inputs = array(
            [fill(process_value(inp, self.bpc), self.dim) for inp in training_inputs]
        )
        training_outputs = array(
            [fill(process_value(out, self.bpc), self.outs) for out in training_outputs]
        )
        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            # siphon the training data via  the neuron
            output = self.think(training_inputs)

            # computing error rate for back-propagation
            error = training_outputs - output

            # performing weight adjustments
            adjustments = dot(training_inputs.T, error * sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        # passing the inputs via the neuron to get output
        if type(inputs) != ndarray:
            inputs = fill(process_value(inputs, self.bpc), self.dim)
        inputs = array(inputs)
        output = sigmoid(dot(inputs, self.synaptic_weights))
        return output

    def addInput(self, start=0):
        if start == None:
            start = sum([sum(x) for x in self.synaptic_weights.tolist()]) / len(
                self.synaptic_weights
            )
        self.synaptic_weights = array(
            [[start] * self.outs] + self.synaptic_weights.tolist()
        )
        self.dim += 1
        self.max_input_length += 1

    def addOutput(self, start=0):
        if start == None:
            start = sum([sum(x) for x in self.synaptic_weights.tolist()]) / len(
                self.synaptic_weights
            )
        for i in range(len(self.synaptic_weights)):
            self.synaptic_weights[i] = array(self.synaptic_weights.tolist() + [start])
        self.outs += 1
        self.max_output_length += 1

    def setInOut(self, inputs, outputs):
        if inputs > self.max_input_length:
            for i in range(inputs - self.max_input_length):
                self.addInput()
        elif inputs < self.max_input_length:
            self.synaptic_weights = self.synaptic_weights[1:]
        if outputs > self.max_output_length:
            for i in range(inputs - self.max_output_length):
                self.addOutput()
        elif outputs < self.max_output_length:
            for i in range(len(self.synaptic_weights)):
                self.synaptic_weights[i] = array(self.synaptic_weights.tolist()[:-1])
        self.max_input_length = self.dim = inputs
        self.max_output_length = self.outs = outputs


if __name__ == "__main__":
    x = NeuralNetwork(32, 32)
    x.train(
        [process_value("hi"), process_value("hey")],
        [process_value("hey"), process_value("hi")],
        5,
    )
    print(decode(x.think(process_value("hi"))))

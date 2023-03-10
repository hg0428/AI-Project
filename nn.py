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


def process_value(x, bits_per_character=8):
    if type(x) == str:
        return [
            int(i)
            for i in "".join([format(ord(i), f"0{bits_per_character}b") for i in x])
        ]
    elif type(x) == int:
        return [int(i) for i in format(x, "b")]
    elif type(x) == list:
        return x
    elif type(x) == array:
        return x
    elif type(x) == ndarray:
        return x


def decode(data, bits_per_character=8):
    confidence = 0
    for i in data:
        if i < 0.5: 
            confidence += 1 - i
        else:
            confidence += i
    confidence /= len(data)
    out = [round(x) for x in data]
    bytes = [
        out[x : x + bits_per_character] for x in range(0, len(out), bits_per_character)
    ]
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
    return string, confidence


class NeuralNetwork:
    def __init__(
        self,
        input_length=4,
        output_length=1,
        bits_per_character=8,
        layers = 1,
        synaptic_weights=None,
        save_funct=None
    ):
        self.layers = layers
        self.input_length = input_length
        self.output_length = output_length
        self.bits_per_character = bits_per_character
        self.save_funct = save_funct
        if synaptic_weights == None:
            # seeding for random number generation
            random.seed(1)

            # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
            self.synaptic_weights = [2 * random.random((input_length, output_length)) - 1] * layers
        else:
            self.synaptic_weights = array(synaptic_weights)
    def __del__(self):
        if self.save_funct:
            self.save_funct()
    def adjust(self, training_inputs, training_outputs):
        # siphon the training data via  the neuron
        output = self.think(training_inputs)

        # computing error rate for back-propagation
        error = training_outputs - output

        # performing weight adjustments
        for i in range(len(self.synaptic_weights)):
            x = output
            for _ in range(len(self.synaptic_weights) - i):
                x = sigmoid_derivative(output)
            adjustments = dot(training_inputs.T, error * x)
    
            self.synaptic_weights[i] += adjustments
        return output

    def train(self, training_inputs, training_outputs, training_iterations):
        training_inputs = array(
            [
                fill(process_value(inp, self.bits_per_character), self.input_length)
                for inp in training_inputs
            ]
        )
        training_outputs = array(
            [
                fill(process_value(out, self.bits_per_character), self.output_length)
                for out in training_outputs
            ]
        )
        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            # siphon the training data via  the neuron
            output = self.think(training_inputs)
    
            # computing error rate for back-propagation
            error = training_outputs - output
    
            # performing weight adjustments
            for i in range(len(self.synaptic_weights)):
                x = output
                for _ in range(len(self.synaptic_weights) - i):
                    x = sigmoid_derivative(output)
                adjustments = dot(training_inputs.T, error * x)
        
                self.synaptic_weights[i] += adjustments

    def think(self, inputs):
        # passing the inputs via the neuron to get output
        if type(inputs) != ndarray:
            inputs = fill(
                process_value(inputs, self.bits_per_character), self.input_length
            )
        inputs = array(inputs)
        x = inputs
        for i in range(len(self.synaptic_weights)):
            x = sigmoid(dot(inputs, self.synaptic_weights[i]))
        output = x
        return output

    def addInput(self, start=0):
        if start == None:
            start = sum([sum(x) for x in self.synaptic_weights.tolist()]) / len(
                self.synaptic_weights
            )
        for l in range(len(self.synaptic_weights)):
            self.synaptic_weights[l] = array(
                [[start] * self.output_length] + self.synaptic_weights[l].tolist()
            )
        self.input_length += 1

    def addOutput(self, start=0):
        if start == None:
            start = sum([sum(x) for x in self.synaptic_weights.tolist()]) / len(
                self.synaptic_weights
            )
        for l in range(len(self.synaptic_weights)):
            for i in range(len(self.synaptic_weights)):
                self.synaptic_weights[l][i] = array(self.synaptic_weights[l].tolist() + [start])
        self.output_length += 1

    def setInOut(self, inputs, outputs):
        if inputs > self.input_length:
            for i in range(inputs - self.input_length):
                self.addInput()
        elif inputs < self.input_length:
            for l in range(len(self.synaptic_weights)):
                self.synaptic_weights[l] = self.synaptic_weights[l][1:]
        if outputs > self.max_output_length:
            for i in range(inputs - self.max_output_length):
                self.addOutput()
        elif outputs < self.max_output_length:
            for l in range(len(self.synaptic_weights)):
                for i in range(len(self.synaptic_weights[l])):
                    self.synaptic_weights[l][i] = array(self.synaptic_weights[l].tolist()[:-1])
        self.input_length = inputs
        self.output_length = outputs


if __name__ == "__main__":
    x = NeuralNetwork(32, 32, 8, 2)
    x.train(
        [process_value("hi"), process_value("hey")],
        [process_value("hey"), process_value("hi")],
        20,
    )
    print(decode(x.think(process_value("hi"))))

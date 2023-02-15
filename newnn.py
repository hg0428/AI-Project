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


def gradient_descent(gradient, start, learn_rate, n_iter):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        vector += diff
    return vector


class Layer:
    def __input__(alg_type, opts):
        if alg_type == 'sigmoid':
            this.function = sigmoid
            this.derivative = sigmoid_derivative
        elif alg_type == 'gradient_descent':
            this.function = lambda x: gradient_descent()




class NeuralNetwork:
    def __init__(
        self,
        layers=[8,8],
        preproccessor=lambda x: process_value(x, 8),
        synaptic_weights=None,
        save_funct=None
    ):
        self.save_funct = save_funct
        self.type = "NN"
        self.layers = layers
        self.preproccessor = preproccessor
        if synaptic_weights == None:
            # seeding for random number generation
            random.seed(1)

            # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
            self.synaptic_weights = []
            for i in range(len(layers)-1):
                self.synaptic_weights.append(2 * random.random((layers[i], layers[i+1])) - 1)
            print(self.synaptic_weights)
        else:
            self.synaptic_weights = [array(synaptic_weights)]
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
                fill(self.preproccessor(inp), self.layers[0])
                for inp in training_inputs
            ]
        )
        training_inputsT = training_inputs.T
        training_outputs = array(
            [
                fill(self.preproccessor(out), self.layers[-1])
                for out in training_outputs
            ]
        )
        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            for i in range(len(self.synaptic_weights)):
                # siphon the training data via  the neuron
                output = self.think(training_inputs, 0, i)
        
                # computing error rate for back-propagation
                error = training_outputs - output
                for _ in range(len(self.synaptic_weights)-i-1):

                    # performing weight adjustments
                    x = sigmoid_derivative(output)

                    adjustments = dot(training_inputsT, error * x)
    
                self.synaptic_weights[len(self.synaptic_weights)-i-1] += adjustments

    def think(self, inputs, start_layer=0, end_layer=-1):
        # passing the inputs via the neuron to get output
        weights = self.synaptic_weights[start_layer]
        if type(inputs) != ndarray:
            inputs = fill(
                self.preproccessor(inputs), self.layers[0]
            )
            inputs = array(inputs)
        x = inputs
        for i in range(len(weights)):
            x = sigmoid(dot(x, weights[i]))
        output = x
        print(output)
        return output

    def setLayers(layers):
        for i in range(len(layers) - 1):
            if i >= len(self.layers):
                self.synaptic_weights.append(2 * random.random((layers[i], layers[i + 1])) - 1)
            else:
                if inputs > self.input_length:
                    for _ in range(inputs - self.input_length):
                        for x in range(len(self.synaptic_weights)):
                            self.synaptic_weights[l] = array(
                                [[0] * layers[i + 1]] + self.synaptic_weights[i].tolist()
                            )
                elif inputs < self.input_length:
                    self.synaptic_weights[i] = self.synaptic_weights[i][1:]
                if outputs > self.max_output_length:
                    for _ in range(inputs - self.max_output_length):
                        for x in range(len(self.synaptic_weights[i])):
                            self.synaptic_weights[i][x] = array(self.synaptic_weights[i][x].tolist() + [0])
                elif outputs < self.max_output_length:
                    for x in range(len(self.synaptic_weights[i])):
                        self.synaptic_weights[i][x] = array(self.synaptic_weights[i].tolist()[:-1])

        self.layers = layers



if __name__ == "__main__":
    t = NeuralNetwork([2, 3, 2])
    [
        array([
            [3],
            [3]
        ]),
        array([
            [2],
            [2],
            [2]
        ])
    ]
    t.train(
        [[0, 1], [1, 0], [1, 1], [0, 0]],
        [[1, 1], [1, 1], [1, 1], [0, 0]],
        50
    )
    print(t.think([1, 1]))
    # x = NeuralNetwork([32, 40, 32])
    # x.train(
    #     [process_value("hi"), process_value("hey")],
    #     [process_value("hey"), process_value("hi")],
    #     20,
    # )
    # print(decode(x.think(process_value("hi"))))

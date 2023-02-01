import nn
import numpy as np


input_length = 5
starting_weights = None
# neural_network = nn.NeuralNetwork(input_length, starting_weights)


def fill(l, length, null=0.5, reverse=False):
    if len(l) > length:
        return l[len(l) - length :]
    for _ in range(length - len(l)):
        if reverse:
            l = [null] + l
        else:
            l.append(null)
    return l


# print(fill("hello", 3))


def process_value(x, bpc=8):
    if type(x) == str:
        return [int(i) for i in "".join([format(ord(i), f"0{bpc}b") for i in x])]
    elif type(x) == int:
        return [int(i) for i in format(x, "b")]
    elif type(x) == list and all(isinstance(i, int) for i in x):
        return x


def decode(data, bpc=8):
    out = [round(x) for x in data]
    bytes = [out[x : x + bpc] for x in range(0, len(out), bpc)]
    strbytes = ["".join([str(i) for i in x]) for x in bytes]
    chrs = [int(x, 2) for x in strbytes]
    string = ""
    for x in chrs:
        try:
            string += chr(x)
        except:
            string += chr(0)
    return string


class DeepLearningModel:
    def __init__(
        self,
        max_input_length,
        max_output_length,
        fill_value=0.5,
        savefunct=None,
        bytes_per_character=8,
    ):
        self.layers = []
        self.savefunct = savefunct
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.fill_value = fill_value
        self.bytes_per_character = bytes_per_character
        self.layers.extend(
            nn.NeuralNetwork(max_input_length) for _ in range(max_output_length)
        )

    def train(self, inputs, outputs, times=2000):
        inputs = np.array(
            [
                fill(
                    process_value(input, self.bytes_per_character),
                    self.max_input_length,
                    self.fill_value,
                    reverse=True,
                )
                for input in inputs
            ]
        )
        outputs = [
            fill(
                process_value(output, self.bytes_per_character),
                self.max_output_length,
                self.fill_value,
            )
            for output in outputs
        ]
        all_outputs = [
            np.array([[output[i] for output in outputs]]).T
            for i in range(self.max_output_length)
        ]
        for _ in range(times):
            for i in range(self.max_output_length):
                self.layers[i].adjust(inputs, all_outputs[i])

    def __del__(self):
        if self.savefunct:
            self.savefunct()

    def think(self, input):
        outputs = []
        input = fill(
            process_value(input, self.bytes_per_character),
            self.max_input_length,
            self.fill_value,
            reverse=True,
        )
        for i in range(self.max_output_length):
            x = self.layers[i].think(input).tolist()
            if type(x) == list:
                x = x[0]
            outputs.append(x)
        return outputs

    def addLayers(self, n=1):
        self.max_output_length += n
        for _ in range(n):
            self.layers = [nn.NeuralNetwork(self.max_input_length)] + self.layers

    def addInputs(self, n=1):
        self.max_input_length += n
        for layer in self.layers:
            for _ in range(n):
                layer.addInput()

    def setInOut(self, inputs, outputs):
        if inputs > self.max_input_length:
            self.addInputs(inputs - self.max_input_length)
        elif inputs < self.max_input_length:
            for i in self.layers:
                i.synaptic_weights = i.synaptic_weights[
                    len(i.synaptic_weights) - inputs :
                ]
        if outputs > self.max_output_length:
            self.addLayers(inputs - self.max_output_length)
        elif outputs < self.max_output_length:
            self.layers = self.layers[:outputs]
        self.max_input_length = inputs
        self.max_output_length = outputs


# print(process_value('Hello World!'), decode(process_value('Hello World!')))

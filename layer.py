"""A layer of the neural network"""
import perceptron
import activation
from typing import Type

class Layer:
    def __init__(self, number_perceptrons: int, activation_func: Type[activation.Activation]):
        self.number_perceptrons = number_perceptrons
        self.activation_func = activation_func
        self.perceptrons = self.createLayer()

    def createLayer(self):
        layer = []
        for i in range(0, self.number_perceptrons):
            layer.append(perceptron.Perceptron(0, self.activation_func, [], []))
        return layer


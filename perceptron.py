"""This file contains the code for the perceptron, the smallest building block of a neural network"""
import activation
from typing import Type

class Synapse:
    def __init__(self, weight: float):
        self.weight = weight
        self.output = 0

    def compute(self, x):
        self.output = self.weight*x

class Perceptron:
    def __init__(self, value: float, activation_func: Type[activation.Activation], receiving_synapses: [Synapse], sending_synapses: [Synapse]):
        self.value = value
        self.activation_func = activation_func
        self.receiving_synapses = receiving_synapses
        self.sending_synapses = sending_synapses

    def compute(self):
        sigma = 0
        if not self.receiving_synapses:
            sigma = self.value
        else:
            for synapse in self.receiving_synapses:
                sigma += synapse.output

        # Apply activation function to sum of values
        self.value = self.activation_func.compute(sigma)

        # If there are no synapses to send to, return the value
        if not self.sending_synapses:
            return self.value
        else:
            # Pass this value to the sending synapses
            for synapse in self.sending_synapses:
                synapse.compute(self.value)

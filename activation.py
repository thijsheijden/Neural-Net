"""This file contains all activation functions which can be used"""
import math

# Base activation class
class Activation:
    def __init__(self):
        pass

    def compute(self, x):
        pass

# Sigmoid activation function
class Sigmoid(Activation):
    def __init__(self):
        Activation.__init__(self)

    def compute(self, x):
        return 1/(1+math.exp(-x))

# Input activation function, which just returns the value
class InputActivation(Activation):
    def __init__(self):
        Activation.__init__(self)

    def compute(self, x):
        return x


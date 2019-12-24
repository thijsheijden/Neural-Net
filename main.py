"""Main file which contains the built neural net"""
import layer
import neural_net
import activation
import pandas

# Data loading and pre-processing
df = pandas.read_csv("data.csv")

# Input layer
input_layer = layer.Layer(4, activation.InputActivation)

# Hidden layer
hidden_layer = layer.Layer(12, activation.Sigmoid)

# Output layer
output_layer = layer.Layer(2, activation.Sigmoid)

layers = [input_layer, hidden_layer, output_layer]

neural_net = neural_net.Neural_Net(layers, 0.2, df, 25)
print("Starting training of neural net")
neural_net.train()

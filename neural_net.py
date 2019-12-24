"""Contains the main neural net class"""
import perceptron
import layer
import random
import pandas

class Neural_Net:
    def __init__(self, layers: [layer.Layer], dropout: float, data: pandas.DataFrame, epochs: int):
        self.layers = layers
        self.dropout = dropout
        self.data = data
        self.epochs = epochs
        self.createSynapses()

    def createSynapses(self):
        synapses = []
        for i in range(0, len(self.layers)):
            current_layer = i
            next_layer = i+1

            # There is no next layer, so this layer gets no sending synapses
            if next_layer == len(self.layers):
                return

            # Loop over the perceptrons in the current layer
            for p_current in self.layers[current_layer].perceptrons:
                # Loop over the perceptrons in the next layer
                for p_next in self.layers[next_layer].perceptrons:
                    # Use dropout to see if the synapse should be added
                    if random.random() < self.dropout:
                        pass
                    else:
                        # Create a new synapse with a weight between 0 and 1
                        synapse = perceptron.Synapse(random.random())
                        # Add the synapse to the current perceptron sending synapses and the next perceptron receiving
                        p_current.sending_synapses.append(synapse)
                        p_next.receiving_synapses.append(synapse)

    def train(self):
        """This method starts the training process for the neural network"""
        for i in range(0, self.epochs):
            print("Starting epoch " + str(i))
            self.doEpoch()
            i += 1

    def doEpoch(self):
        """This method starts an epoch, which includes one feed forward iteration and then one back propagation iterations"""
        self.feedForward()
        self.backPropagate()

    def feedForward(self):
        # For every row of data do an iteration of feed forward and back propagation
        for row in self.data:

            i = 0
            # Set values of input neurons to new data
            for input_perceptron in self.layers[0].perceptrons:
                input_perceptron.value = row[i]
                i += 1

    def backPropagate(self):
        pass


"""A simple fully connected neural network built with NumPy only."""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1.0 - np.tanh(z) ** 2


# Each activation is paired with its derivative (needed later for backprop).
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_prime),
    "tanh": (tanh, tanh_prime),
}


class NeuralNetwork:
    """Fully connected network.

        layer_sizes: Number of neurons in each layer, input to output.
            For example [784, 64, 10] is 784 inputs, one hidden layer of
            64 neurons, and 10 outputs.
        activation: "sigmoid" or "tanh".
        seed: Random seed for weight initialization.
    """

    def __init__(self, layer_sizes, activation="sigmoid", seed=0):
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(ACTIVATIONS)}")

        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.activation, self.activation_prime = ACTIVATIONS[activation]

        # One weight matrix and bias vector per connection between layers.
        # weights[l] has shape (n_out, n_in), scaled by 1/sqrt(n_in) so the
        # neurons don't start out saturated.
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.standard_normal((n_out, n_in)) / np.sqrt(n_in)
            for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.zeros((n_out, 1)) for n_out in layer_sizes[1:]]

    def forward(self, X):
        """Run the forward pass.

        X is a batch of N examples, shape (N, 28, 28) or (N, 784). Returns
        the output layer's activations, shape (N, n_outputs).
        """
        # Flatten each image and put examples in columns: (n_inputs, N).
        a = X.reshape(len(X), -1).T

        # Keep each layer's pre-activations z and activations a; backprop
        # will need them later.
        self.zs = []
        self.activations = [a]
        for W, b in zip(self.weights, self.biases):
            z = W @ a + b
            a = self.activation(z)
            self.zs.append(z)
            self.activations.append(a)

        return a.T

    def predict(self, X):
        """Predict the class of each example: the index of its largest output."""
        return np.argmax(self.forward(X), axis=1)

    def __repr__(self):
        return f"NeuralNetwork({self.layer_sizes}, activation={self.activation_name!r})"

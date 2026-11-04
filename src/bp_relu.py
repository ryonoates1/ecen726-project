"""Backprop baseline with ReLU hidden layers, for a fair comparison with Forward-Forward.

The group's NeuralNetwork / backprop / CrossbarNetwork use one activation for every
layer, including the output. FF uses ReLU hidden layers, so the BP baseline should too;
only the learning rule should differ between the two. Here the hidden layers are ReLU
and the output layer stays sigmoid, so the same binary cross-entropy loss as
`backprop.py` still applies. Nothing in the group's files is changed.
"""

import numpy as np

from src.crossbar import CrossbarNetwork
from src.network import NeuralNetwork, sigmoid


def relu(z):
    return np.maximum(z, 0.0)


def relu_prime(z):
    # The derivative at exactly z = 0 is undefined; 0 is the usual choice.
    return (z > 0).astype(z.dtype)


class ReLUNetwork(NeuralNetwork):
    """Fully connected network with ReLU hidden layers and a sigmoid output layer.

        layer_sizes: Number of neurons in each layer, input to output,
            e.g. [784, 64, 64, 10].
        seed: Random seed for weight initialization.
    """

    def __init__(self, layer_sizes, seed=0):
        # Reuse the parent's weight/bias layout and random draws: weights[l] is
        # (n_out, n_in), biases[l] is (n_out, 1). "sigmoid" here only passes the
        # parent's check; the activations are replaced right below.
        super().__init__(layer_sizes, activation="sigmoid", seed=seed)
        self.activation_name = "relu"
        self.activation, self.activation_prime = relu, relu_prime

        # He initialization: ReLU zeros about half its inputs, so scale the
        # weights by sqrt(2/n_in) instead of sqrt(1/n_in) to keep the variance
        # of z roughly constant from layer to layer.
        self.weights = [W * np.sqrt(2.0) for W in self.weights]

    def forward(self, X):
        """Run the forward pass.

        Args:
            X: Batch of N examples, shape (N, 28, 28) or (N, 784).

        Returns:
            Output activations, shape (N, n_outputs). net.zs and
            net.activations hold the per-layer values in column layout
            (n_neurons, N), as in NeuralNetwork.
        """
        a = X.reshape(len(X), -1).T
        self.zs = []
        self.activations = [a]
        last = len(self.weights) - 1
        for l, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ a + b
            a = sigmoid(z) if l == last else relu(z)
            self.zs.append(z)
            self.activations.append(a)
        return a.T

    def __repr__(self):
        return f"ReLUNetwork({self.layer_sizes}, hidden='relu', output='sigmoid')"


def _cross_entropy(a, t):
    """Average binary cross-entropy per example; a and t have shape (n_outputs, B)."""
    a = np.clip(a, 1e-12, 1 - 1e-12)  # avoid log(0)
    return -np.mean(np.sum(t * np.log(a) + (1 - t) * np.log(1 - a), axis=0))


def gradients(net, X, y):
    """Loss and its gradients for one batch, by backpropagation.

    Args:
        net: A ReLUNetwork.
        X: Batch of images, shape (B, 28, 28) or (B, 784).
        y: Integer labels, shape (B,).

    Returns:
        (loss, grads_W, grads_b): average loss over the batch and lists of
        gradients with the same shapes as net.weights and net.biases.
    """
    B = len(y)
    t = np.eye(net.layer_sizes[-1])[y].T  # one-hot targets, (n_outputs, B)
    net.forward(X)
    a_out = net.activations[-1]
    loss = _cross_entropy(a_out, t)

    # Output layer: with a sigmoid output and cross-entropy loss the sigmoid
    # derivative cancels, so dLoss/dz = a - t.
    delta = a_out - t
    L = len(net.weights)
    grads_W, grads_b = [None] * L, [None] * L
    for l in range(L - 1, -1, -1):
        grads_W[l] = delta @ net.activations[l].T / B
        grads_b[l] = delta.mean(axis=1, keepdims=True)
        if l > 0:
            # Hidden layer below is ReLU: the error passes only through units
            # that were active (z > 0). Note W^T, the "weight transport" FF avoids.
            delta = (net.weights[l].T @ delta) * relu_prime(net.zs[l - 1])
    return loss, grads_W, grads_b


def check_gradients(net, X, y, n_checks=20, eps=1e-6, seed=0):
    """Compare backprop gradients with central finite differences.

    Args:
        net: A ReLUNetwork (its weights are perturbed and then restored).
        X, y: A small batch of examples and labels.
        n_checks: Number of randomly chosen weights/biases to check per layer.
        eps: Finite-difference step.
        seed: Seed for choosing which entries to check.

    Returns:
        Largest relative error |g_bp - g_fd| / max(|g_bp| + |g_fd|, 1e-6). The
        floor keeps gradients below finite-difference resolution (~1e-10 in
        float64 with eps = 1e-6) from showing a meaningless relative error.
    """
    rng = np.random.default_rng(seed)
    _, grads_W, grads_b = gradients(net, X, y)
    worst = 0.0
    for params, grads in ((net.weights, grads_W), (net.biases, grads_b)):
        for P, G in zip(params, grads):
            for _ in range(n_checks):
                idx = tuple(rng.integers(0, s) for s in P.shape)
                old = P[idx]
                P[idx] = old + eps
                loss_plus = gradients(net, X, y)[0]
                P[idx] = old - eps
                loss_minus = gradients(net, X, y)[0]
                P[idx] = old
                g_fd = (loss_plus - loss_minus) / (2 * eps)
                rel = abs(G[idx] - g_fd) / max(abs(G[idx]) + abs(g_fd), 1e-6)
                worst = max(worst, rel)
    return worst


def backprop_relu(net, X, y, epochs, lr=0.1, batch_size=32, seed=0,
                  X_test=None, y_test=None):
    """Train a ReLUNetwork with mini-batch gradient descent and backpropagation.

    Same schedule, loss and history format as `backprop.backprop(loss="ce")`.

    Args:
        net: A ReLUNetwork.
        X: Training images, shape (N, 28, 28) or (N, 784).
        y: Integer labels, shape (N,).
        epochs: Number of passes over the training data.
        lr: Learning rate.
        batch_size: Number of examples per gradient step.
        seed: Random seed for shuffling the data each epoch.
        X_test, y_test: Optional held-out data; if given, its accuracy is
            recorded after every epoch (it is never trained on).

    Returns:
        (net, history): the trained net (also updated in place) and a dict with
        per-epoch lists "loss", "train_acc" and, if X_test is given, "test_acc".
    """
    rng = np.random.default_rng(seed)
    history = {"loss": [], "train_acc": []}
    if X_test is not None:
        history["test_acc"] = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(X))
        epoch_loss = 0.0

        for start in range(0, len(X), batch_size):
            idx = order[start:start + batch_size]
            loss, grads_W, grads_b = gradients(net, X[idx], y[idx])
            epoch_loss += loss * len(idx)
            for l in range(len(net.weights)):
                net.weights[l] -= lr * grads_W[l]
                net.biases[l] -= lr * grads_b[l]

        train_acc = (net.predict(X) == y).mean()
        history["loss"].append(epoch_loss / len(X))
        history["train_acc"].append(train_acc)
        msg = f"epoch {epoch}/{epochs}: loss {history['loss'][-1]:.4f}, train acc {train_acc:.4f}"
        if X_test is not None:
            test_acc = (net.predict(X_test) == y_test).mean()
            history["test_acc"].append(test_acc)
            msg += f", test acc {test_acc:.4f}"
        print(msg)

    return net, history


class ReLUCrossbarNetwork(CrossbarNetwork):
    """A trained ReLUNetwork with each layer programmed onto a crossbar.

    Programming (G+/G- pair, bias row, noise, clipping) is the group's
    `Crossbar` unchanged; only the activations applied to the column outputs
    differ: ReLU after hidden layers, sigmoid after the last.

        net: Trained ReLUNetwork whose weights are transferred.
        sigma: Programming noise std in uS.
        seed: Seed for the programming noise (None gives a new draw each time).
    """

    def forward(self, X):
        # Crossbar convention: examples are rows, shape (B, n_features).
        a = X.reshape(len(X), -1)
        last = len(self.layers) - 1
        for l, xbar in enumerate(self.layers):
            z = xbar.forward(a)
            # Physics note: the next layer's row voltage is V_READ * a. ReLU
            # outputs are unbounded, so unlike sigmoid (0-0.2 V) this can ask
            # for large read voltages; the simulation is linear and ignores that.
            a = sigmoid(z) if l == last else relu(z)
        return a

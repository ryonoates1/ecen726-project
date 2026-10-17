"""Train a NeuralNetwork with backpropagation and mini-batch gradient descent."""

import numpy as np


def backprop(net, X, y, epochs, loss="ce", lr=0.1, batch_size=32, seed=0,
             X_test=None, y_test=None):
    """Train net with mini-batch gradient descent and backpropagation.

    Args:
        net: A NeuralNetwork.
        X: Training images, shape (N, 28, 28) or (N, 784).
        y: Integer labels, shape (N,).
        epochs: Number of passes over the training data.
        loss: "ce" (cross-entropy, sigmoid only) or "mse" (mean squared error).
        lr: Learning rate.
        batch_size: Number of examples per gradient step.
        seed: Random seed for shuffling the data each epoch.
        X_test, y_test: Optional held-out data; if given, its accuracy is
            recorded after every epoch (it is never trained on).

    Returns:
        (net, history): the trained net (it is also updated in place) and a
        dict with per-epoch lists "loss", "train_acc" and, if X_test is
        given, "test_acc".
    """
    if loss not in ("ce", "mse"):
        raise ValueError('loss must be "ce" or "mse"')
    if loss == "ce" and net.activation_name != "sigmoid":
        # Cross-entropy takes log(a) and log(1 - a), so the outputs must be
        # in (0, 1). tanh outputs go down to -1.
        raise ValueError('cross-entropy needs activation="sigmoid"; use loss="mse" with tanh')

    def loss_fn(a, t):
        """Average loss per example. a and t have shape (n_outputs, B)."""
        if loss == "mse":
            return 0.5 * np.mean(np.sum((a - t) ** 2, axis=0))
        a = np.clip(a, 1e-12, 1 - 1e-12)  # avoid log(0)
        return -np.mean(np.sum(t * np.log(a) + (1 - t) * np.log(1 - a), axis=0))

    def output_delta(a, t, z):
        """dLoss/dz for the output layer."""
        if loss == "mse":
            return (a - t) * net.activation_prime(z)
        # Cross-entropy with sigmoid: the sigmoid derivative cancels out.
        return a - t

    rng = np.random.default_rng(seed)
    n_outputs = net.layer_sizes[-1]
    history = {"loss": [], "train_acc": []}
    if X_test is not None:
        history["test_acc"] = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(X))
        epoch_loss = 0.0

        for start in range(0, len(X), batch_size):
            idx = order[start:start + batch_size]
            B = len(idx)
            t = np.eye(n_outputs)[y[idx]].T  # one-hot targets, (n_outputs, B)

            # Forward pass; net.zs and net.activations are filled in.
            net.forward(X[idx])
            a_out = net.activations[-1]
            epoch_loss += loss_fn(a_out, t) * B

            # Backward pass: delta is dLoss/dz for the current layer,
            # starting at the output and moving back one layer at a time.
            delta = output_delta(a_out, t, net.zs[-1])
            for l in range(len(net.weights) - 1, -1, -1):
                grad_W = delta @ net.activations[l].T / B
                grad_b = delta.mean(axis=1, keepdims=True)
                if l > 0:
                    # Use this layer's weights before they are updated.
                    delta = (net.weights[l].T @ delta) * net.activation_prime(net.zs[l - 1])
                net.weights[l] -= lr * grad_W
                net.biases[l] -= lr * grad_b

        # Accuracy on the full training set with this epoch's final weights.
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

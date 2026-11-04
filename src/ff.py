"""Forward-Forward training (Hinton 2022) of a fully connected ReLU net, NumPy only.

Each layer has its own local objective: make the goodness G = sum_j y_j^2 high for
positive data (image + correct label) and low for negative data (image + wrong label).
No error is ever sent backwards through W^T; every weight update is an outer product
of a layer's own input and a local error term.
"""

import numpy as np

N_CLASSES = 10
NEUTRAL = 0.1  # label value used in all 10 label pixels when the label is unknown


# ---------------------------------------------------------------------------
# Data: putting the label into the image
# ---------------------------------------------------------------------------

def embed_label(X, labels):
    """Overwrite the first 10 pixels of each image with a one-hot label.

    The first 10 pixels (top-left of the border) are 0 in every MNIST image,
    so no image information is lost.

    Args:
        X: Images, shape (N, 28, 28) or (N, 784).
        labels: Integer labels, shape (N,), or None for the neutral label
            (0.1 in each of the 10 pixels).

    Returns:
        New array, shape (N, 784), float64. X is not modified.
    """
    Xe = X.reshape(len(X), -1).astype(np.float64)  # astype copies
    if labels is None:
        Xe[:, :N_CLASSES] = NEUTRAL
    else:
        Xe[:, :N_CLASSES] = 0.0
        Xe[np.arange(len(X)), labels] = 1.0
    return Xe


def random_wrong_labels(y, rng):
    """Draw a label != y for every example, uniformly over the other 9 classes.

    Args:
        y: True labels, shape (N,).
        rng: NumPy Generator.

    Returns:
        Wrong labels, shape (N,).
    """
    # Adding 1..9 (mod 10) to the true label never lands back on it.
    return (y + rng.integers(1, N_CLASSES, size=len(y))) % N_CLASSES


def hard_wrong_labels(goodness, y, rng):
    """Pick wrong labels the network currently finds plausible ("hard negatives").

    Hinton 2022, Sec. 3.3: sample an incorrect label in proportion to how much
    the net likes it, so training focuses on the confusions it still makes.

    Args:
        goodness: Total goodness for every label, shape (N, 10), e.g. from
            FFNetwork.goodness_per_label.
        y: True labels, shape (N,).
        rng: NumPy Generator.

    Returns:
        Wrong labels, shape (N,).
    """
    g = goodness - goodness.max(axis=1, keepdims=True)  # stable softmax
    p = np.exp(g)
    p[np.arange(len(y)), y] = 0.0                        # never pick the true label
    p /= p.sum(axis=1, keepdims=True)
    # Inverse-CDF sampling, one draw per row.
    u = rng.random((len(y), 1))
    return np.minimum((p.cumsum(axis=1) < u).sum(axis=1), N_CLASSES - 1)


# ---------------------------------------------------------------------------
# One layer: forward pass, local loss, hand-derived gradient
# ---------------------------------------------------------------------------

def relu(z):
    return np.maximum(z, 0.0)


def normalize(a, eps=1e-8):
    """Scale every column of a to unit length (no mean subtraction).

    The length of a layer's activity vector is its goodness, so only the
    direction is passed on; otherwise the next layer could just read off the
    previous layer's goodness and learn nothing new.

    Args:
        a: Activities, shape (n, B), examples as columns.
        eps: Avoids division by zero for an all-zero vector.

    Returns:
        Array of the same shape with every column of length ~1.
    """
    return a / (np.sqrt(np.sum(a * a, axis=0, keepdims=True)) + eps)


def _softplus(x):
    return np.logaddexp(0.0, x)       # log(1 + e^x) without overflow


def _sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))  # 1 / (1 + e^-x) without overflow


def layer_loss_and_grads(W, b, x_pos, x_neg, theta):
    """Local FF loss of one layer and its gradient, derived by hand.

    For each example: z = W x + b, y = ReLU(z), G = sum_j y_j^2, and
        L = softplus(-(G_pos - theta)) + softplus(G_neg - theta),
    averaged over the batch. Chain rule (no autograd):
        dL/dG_pos = -sigmoid(-(G_pos - theta)) = -(1 - p_pos)
        dL/dG_neg =  sigmoid(G_neg - theta)    =  p_neg
        dG/dz     =  2 y          (y = 0 wherever z <= 0, so the ReLU mask is built in)
        dL/dW     =  delta x^T    with delta = dL/dG * dG/dz  -> outer product
        dL/db     =  sum of delta over the batch

    Args:
        W: Weights, shape (n_out, n_in).
        b: Biases, shape (n_out, 1).
        x_pos, x_neg: Positive / negative inputs, shape (n_in, B), columns.
        theta: Goodness threshold.

    Returns:
        (loss, grad_W, grad_b, y_pos, y_neg, G_pos, G_neg). y_* have shape
        (n_out, B) and G_* shape (B,), so the caller can feed the next layer.
    """
    B = x_pos.shape[1]
    y_pos = relu(W @ x_pos + b)
    y_neg = relu(W @ x_neg + b)
    G_pos = np.sum(y_pos ** 2, axis=0)
    G_neg = np.sum(y_neg ** 2, axis=0)

    loss = np.mean(_softplus(-(G_pos - theta)) + _softplus(G_neg - theta))

    dG_pos = -_sigmoid(-(G_pos - theta)) / B   # 1/B from the batch mean
    dG_neg = _sigmoid(G_neg - theta) / B
    delta_pos = dG_pos * 2.0 * y_pos           # (n_out, B); dG_* broadcasts over rows
    delta_neg = dG_neg * 2.0 * y_neg

    grad_W = delta_pos @ x_pos.T + delta_neg @ x_neg.T
    grad_b = (delta_pos.sum(axis=1) + delta_neg.sum(axis=1))[:, None]
    return loss, grad_W, grad_b, y_pos, y_neg, G_pos, G_neg


def check_layer_gradients(W, b, x_pos, x_neg, theta, n_checks=30, eps=1e-6, seed=0):
    """Compare layer_loss_and_grads with central finite differences.

    Args:
        W, b, x_pos, x_neg, theta: As in layer_loss_and_grads. W and b are
            perturbed in place and restored.
        n_checks: Number of random entries of W (and of b) to check.
        eps: Finite-difference step.
        seed: Seed for choosing the entries.

    Returns:
        Largest relative error |g - g_fd| / max(|g| + |g_fd|, 1e-6). The 1e-6
        floor matters: in float64 the finite difference of an O(1) loss is
        only resolved to ~1e-16 / eps = 1e-10, so a gradient of ~1e-9 (e.g. a
        unit active for one example) would otherwise show a meaningless
        relative error near 1.
    """
    rng = np.random.default_rng(seed)
    _, gW, gb = layer_loss_and_grads(W, b, x_pos, x_neg, theta)[:3]
    worst = 0.0
    for P, G in ((W, gW), (b, gb)):
        for _ in range(n_checks):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            old = P[idx]
            P[idx] = old + eps
            l_plus = layer_loss_and_grads(W, b, x_pos, x_neg, theta)[0]
            P[idx] = old - eps
            l_minus = layer_loss_and_grads(W, b, x_pos, x_neg, theta)[0]
            P[idx] = old
            g_fd = (l_plus - l_minus) / (2 * eps)
            worst = max(worst, abs(G[idx] - g_fd) / max(abs(G[idx]) + abs(g_fd), 1e-6))
    return worst


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class FFNetwork:
    """Stack of ReLU layers trained with Forward-Forward.

    There is no output layer: the class enters through the label pixels, and a
    digit is classified by which embedded label makes the net "happiest".

        layer_sizes: Input size then hidden sizes, e.g. [784, 500, 500].
        theta: Goodness threshold, per layer. A number is used for every
            layer, a list gives one per layer, None uses each layer's width
            (Hinton's suggestion). With unit-length inputs and G = sum y^2,
            theta = width saturates the loss; 5 was best on validation for
            784-500-500 (1, 2, 5 tried).
        seed: Random seed for weight initialization.
    """

    def __init__(self, layer_sizes, theta=5.0, seed=0):
        self.layer_sizes = layer_sizes
        n_layers = len(layer_sizes) - 1
        if theta is None:
            self.theta = [float(n) for n in layer_sizes[1:]]
        elif np.isscalar(theta):
            self.theta = [float(theta)] * n_layers
        else:
            self.theta = [float(t) for t in theta]

        # Same layout as NeuralNetwork: weights[l] is (n_out, n_in), biases[l]
        # is (n_out, 1). Every layer's input is a unit-length vector, so with
        # i.i.d. weights of variance s^2 each z_j = w_j . x has variance s^2 and
        # E[ReLU(z_j)^2] = s^2 / 2. Choosing s^2 = 2 / n_out makes the initial
        # goodness E[G] = n_out * s^2 / 2 = 1 for any width.
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / n_out)
            for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.zeros((n_out, 1)) for n_out in layer_sizes[1:]]

    def layer_outputs(self, X_embedded):
        """Forward pass, returning each layer's activities and goodness.

        Args:
            X_embedded: Inputs with the label already embedded, shape (N, 784).

        Returns:
            (ys, Gs): lists over layers of activities, shape (n_out, N), and
            goodness, shape (N,).
        """
        x = normalize(X_embedded.T)  # columns, unit length
        ys, Gs = [], []
        for W, b in zip(self.weights, self.biases):
            y = relu(W @ x + b)
            ys.append(y)
            Gs.append(np.sum(y ** 2, axis=0))
            x = normalize(y)         # pass on direction only
        return ys, Gs

    def goodness_per_label(self, X, skip_first=False, batch_size=2000):
        """Total goodness of every example under each of the 10 labels.

        Args:
            X: Images, shape (N, 28, 28) or (N, 784).
            skip_first: If True, leave out the first layer's goodness
                (Hinton does this for deeper nets).
            batch_size: Examples per forward pass, to bound memory.

        Returns:
            Array of shape (N, 10): column k is the summed goodness with label k.
        """
        N = len(X)
        out = np.zeros((N, N_CLASSES))
        first = 1 if skip_first else 0
        for start in range(0, N, batch_size):
            Xb = X[start:start + batch_size]
            for k in range(N_CLASSES):
                _, Gs = self.layer_outputs(embed_label(Xb, np.full(len(Xb), k)))
                out[start:start + len(Xb), k] = sum(Gs[first:])
        return out

    def predict(self, X, skip_first=False):
        """Classify each image as the label with the largest total goodness."""
        return np.argmax(self.goodness_per_label(X, skip_first), axis=1)

    def __repr__(self):
        return f"FFNetwork({self.layer_sizes}, theta={self.theta})"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class _Updater:
    """Turns a gradient into a parameter step: SGD with momentum, or Adam.

    Adam (Kingma & Ba 2015) divides each parameter's step by a running RMS of
    its own gradient, so every weight moves at a similar rate whatever its
    gradient scale. FF gradients 2 y x^T are tiny for unit-length inputs and
    vary a lot between units, which makes plain SGD's stable range narrow.
    """

    def __init__(self, params, method, lr, momentum=0.9, beta2=0.999, eps=1e-8):
        if method not in ("sgd", "adam"):
            raise ValueError('optimizer must be "sgd" or "adam"')
        self.method, self.lr, self.m, self.beta2, self.eps = method, lr, momentum, beta2, eps
        self.v1 = [np.zeros_like(p) for p in params]  # momentum / Adam 1st moment
        self.v2 = [np.zeros_like(p) for p in params]  # Adam 2nd moment
        self.t = [0] * len(params)

    def step(self, i, grad):
        """Return the change to apply to parameter i for gradient grad."""
        if self.method == "sgd":
            self.v1[i] = self.m * self.v1[i] - self.lr * grad
            return self.v1[i]
        self.t[i] += 1
        self.v1[i] = self.m * self.v1[i] + (1 - self.m) * grad
        self.v2[i] = self.beta2 * self.v2[i] + (1 - self.beta2) * grad ** 2
        m_hat = self.v1[i] / (1 - self.m ** self.t[i])         # bias correction
        v_hat = self.v2[i] / (1 - self.beta2 ** self.t[i])
        return -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train_ff(net, X, y, epochs, lr=0.003, optimizer="adam", momentum=0.9, batch_size=100,
             negatives="random", skip_first=False, seed=0, X_test=None, y_test=None):
    """Train an FFNetwork with Forward-Forward, all layers on every batch.

    For each batch, the positive (correct label) and negative (wrong label)
    versions of the images go forward through the net. Each layer updates its
    own W and b from its own loss only, then passes the normalized activities
    of both versions to the next layer. Nothing travels backwards.

    Args:
        net: An FFNetwork (updated in place).
        X: Training images, shape (N, 28, 28) or (N, 784).
        y: Integer labels, shape (N,).
        epochs: Number of passes over the training data.
        lr: Learning rate.
        optimizer: "adam" or "sgd" (SGD with momentum).
        momentum: SGD momentum, or Adam's beta1.
        batch_size: Examples per update (each gives one positive and one negative).
        negatives: "random" (uniform wrong label) or "hard" (wrong label
            sampled from the net's own goodness, recomputed every epoch;
            epoch 1 uses random labels).
        skip_first: Passed to predict: leave layer 1 out of the goodness sum.
        seed: Seed for shuffling and for drawing negative labels.
        X_test, y_test: Optional held-out data whose accuracy is recorded
            after every epoch (never trained on). Use the validation set while
            tuning.

    Returns:
        (net, history): history has per-epoch lists "loss" (sum of the layer
        losses), "layer_loss" (one list per epoch), "train_acc" and, if X_test
        is given, "test_acc".
    """
    if negatives not in ("random", "hard"):
        raise ValueError('negatives must be "random" or "hard"')
    rng = np.random.default_rng(seed)
    n_layers = len(net.weights)
    # Parameters in the order [W_0, b_0, W_1, b_1, ...] for the updater.
    upd = _Updater([p for W, b in zip(net.weights, net.biases) for p in (W, b)],
                   optimizer, lr, momentum)
    history = {"loss": [], "layer_loss": [], "train_acc": []}
    if X_test is not None:
        history["test_acc"] = []
    train_goodness = None  # (N, 10), refreshed at the end of every epoch

    for epoch in range(1, epochs + 1):
        if negatives == "hard" and train_goodness is not None:
            y_neg_all = hard_wrong_labels(train_goodness, y, rng)
        else:
            y_neg_all = random_wrong_labels(y, rng)

        order = rng.permutation(len(X))
        layer_loss = np.zeros(n_layers)

        for start in range(0, len(X), batch_size):
            idx = order[start:start + batch_size]
            x_pos = normalize(embed_label(X[idx], y[idx]).T)        # (784, B)
            x_neg = normalize(embed_label(X[idx], y_neg_all[idx]).T)

            for l in range(n_layers):
                loss, gW, gb, y_pos, y_neg, _, _ = layer_loss_and_grads(
                    net.weights[l], net.biases[l], x_pos, x_neg, net.theta[l])
                layer_loss[l] += loss * len(idx)
                net.weights[l] += upd.step(2 * l, gW)
                net.biases[l] += upd.step(2 * l + 1, gb)
                # The next layer sees only the direction of this layer's activity.
                x_pos, x_neg = normalize(y_pos), normalize(y_neg)

        train_goodness = net.goodness_per_label(X, skip_first)
        train_acc = (np.argmax(train_goodness, axis=1) == y).mean()
        layer_loss /= len(X)
        history["loss"].append(layer_loss.sum())
        history["layer_loss"].append(layer_loss.tolist())
        history["train_acc"].append(train_acc)
        msg = (f"epoch {epoch}/{epochs}: loss {layer_loss.sum():.4f} "
               f"({', '.join(f'{v:.3f}' for v in layer_loss)}), train acc {train_acc:.4f}")
        if X_test is not None:
            test_acc = (net.predict(X_test, skip_first) == y_test).mean()
            history["test_acc"].append(test_acc)
            msg += f", test acc {test_acc:.4f}"
        print(msg)

    return net, history

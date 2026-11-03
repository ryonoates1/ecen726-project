import numpy as np

G_MIN = 0.1      # uS
G_MAX = 150.0    # uS
SIGMA = 0.1      # uS, programming noise (std of a Gaussian per device)
V_READ = 0.2     # V, read voltage for an input activation of 1


def vmm_reference(G, v):
    """Crossbar VMM for one input vector, written out device by device.

    G has shape (n_rows, n_cols) in uS and v has shape (n_rows,) in V.
    Returns the column currents, shape (n_cols,), in uA.
    """
    n_rows, n_cols = G.shape
    I = np.zeros(n_cols)
    for j in range(n_cols):            # each column wire
        for i in range(n_rows):        # each device on that column
            I[j] += G[i, j] * v[i]     # Ohm's law, summed by KCL
    return I


def vmm(G, V):
    """Crossbar VMM for a batch of inputs: the same sum as vmm_reference.

    G has shape (n_rows, n_cols) and V has shape (B, n_rows). Returns the
    column currents, shape (B, n_cols). The loops of vmm_reference are too
    slow for 10k test images, so the sum over i is done with einsum.
    """
    return np.einsum("bi,ij->bj", V, G)


class Crossbar:
    """One layer's weights and bias programmed into a differential crossbar.

    The target conductances come from the ideal weights; programming then
    adds Gaussian noise to every device and clips to the device range.

        W: Weight matrix, shape (n_out, n_in) as in NeuralNetwork.
        b: Bias vector, shape (n_out, 1).
        sigma: Programming noise std in uS (0 gives an ideal transfer).
        rng: NumPy Generator used for the noise.
    """

    def __init__(self, W, b, sigma=SIGMA, rng=None, g_min=G_MIN, g_max=G_MAX):
        rng = np.random.default_rng() if rng is None else rng
        self.g_min, self.g_max, self.sigma = g_min, g_max, sigma

        # Crossbar layout: rows are inputs (plus one bias row), columns are
        # outputs, so the weights are transposed: shape (n_in + 1, n_out).
        W_aug = np.vstack([W.T, b.T])

        # Scale so the largest |weight| in this layer uses the full range.
        self.g_per_w = (g_max - g_min) / np.abs(W_aug).max()   # uS per unit weight

        # Positive weights go on G+, negative ones on G-; the other device
        # of the pair sits at G_MIN.
        self.G_pos_target = g_min + self.g_per_w * np.maximum(W_aug, 0)
        self.G_neg_target = g_min + self.g_per_w * np.maximum(-W_aug, 0)

        # Program the devices: every device gets its own noise sample.
        self.G_pos = self._program(self.G_pos_target, rng)
        self.G_neg = self._program(self.G_neg_target, rng)

    def _program(self, G_target, rng):
        G = G_target + rng.normal(0.0, self.sigma, G_target.shape)#This adds the programming error as well as clipping
        return np.clip(G, self.g_min, self.g_max)

    def effective_weights(self):
        """The weights the programmed devices actually represent, shape (n_out, n_in)."""
        W_aug = (self.G_pos - self.G_neg) / self.g_per_w
        return W_aug[:-1].T

    def forward(self, a):
        """Pre-activations z for activations a of shape (B, n_in)."""
        # Activations become row voltages; the bias row is held at V_READ.
        V = V_READ * np.hstack([a, np.ones((len(a), 1))])
        I = vmm(self.G_pos, V) - vmm(self.G_neg, V)   # differential column current, uA
        # Convert the current back to weight units.
        return I / (V_READ * self.g_per_w)


class CrossbarNetwork:
    """A trained NeuralNetwork with each layer programmed onto a crossbar.

    Each new CrossbarNetwork is one independent programming of the chip.

        net: Trained NeuralNetwork whose weights are transferred.
        sigma: Programming noise std in uS.
        seed: Seed for the programming noise (None gives a new draw each time).
    """

    def __init__(self, net, sigma=SIGMA, seed=None):
        rng = np.random.default_rng(seed)
        self.activation = net.activation
        self.layers = [Crossbar(W, b, sigma, rng) for W, b in zip(net.weights, net.biases)]

    def forward(self, X):
        a = X.reshape(len(X), -1)
        for xbar in self.layers:
            a = self.activation(xbar.forward(a))
        return a

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

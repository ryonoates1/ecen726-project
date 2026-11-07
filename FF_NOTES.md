# Forward-Forward (FF) on MNIST

## Files

| File | Contents |
|---|---|
| `src/ff.py` | FF: label embedding, negatives, local loss + hand-derived gradient, `FFNetwork`, `train_ff` |
| `src/bp_relu.py` | BP baseline, ReLU hidden + sigmoid output: `ReLUNetwork`, `backprop_relu`, `ReLUCrossbarNetwork` |
| `ff.ipynb` | Experiments and figures |
| `requirements.txt` | `pywinpty` restricted to Windows (`sys_platform == "win32"`) |

Existing files in `src/` and `main.ipynb` are unchanged.

## Results (test set evaluated once)

| Model | Hidden | Params | Optimizer | Val | Test |
|---|---|---|---|---|---|
| BP, sigmoid | 64-64 | 55,050 | SGD 0.1, bs 32 | 97.13 % | 97.52 % |
| BP, ReLU | 64-64 | 55,050 | SGD 0.1, bs 32 | 97.49 % | 97.78 % |
| FF | 64-64 | 54,400 | Adam 0.003, bs 100 | 96.09 % | — |
| BP, ReLU | 500-500 | 648,010 | SGD 0.1, bs 32 | 98.22 % | 98.40 % |
| BP, ReLU | 500-500 | 648,010 | Adam 0.003, bs 100 | 98.05 % | — |
| FF | 500-500 | 643,000 | Adam 0.003, bs 100 | 97.90 % | 97.91 % |

Epochs: 60, except the two 64-64 BP runs (50). FF: θ = 5, hard negatives. Hyperparameters chosen on the validation set.

- BP has 5,010 more parameters: its 10-unit output layer. FF reads the class through the label pixels inside the 784 inputs; hidden layers are identical.
- With the same optimizer (Adam 0.003, bs 100), BP still leads FF (98.05 vs 97.90 % val), so the gap is not due to the optimizer.
- FF needs width: the gap to ReLU BP is 0.3 points at 500-500 and 1.4 points at 64-64 (θ not re-tuned for 64-64).

## Forward-Forward algorithm (`src/ff.py`)

Notation: layer $l$ has weights $W$ of shape $(n_\text{out}, n_\text{in})$, bias $b$ of shape $(n_\text{out}, 1)$,
threshold $\theta$; a batch has $B$ examples; $\sigma(u) = 1/(1+e^{-u})$, $\operatorname{softplus}(u) = \log(1+e^{u})$.

**1. Label embedding** (`embed_label`). Pixels 0–9 of the flattened image are always 0 in MNIST and are overwritten:

$$\tilde{x} = [\,e_k,\ \text{pixels } 10..783\,],\qquad e_k = \text{one-hot}(k),\qquad \text{neutral label: all 10 entries } = 0.1$$

Positive example: $k = y$ (true label). Negative example: same image, $k \ne y$.

**2. Input normalization** (`normalize`). Every layer input has unit length (no mean subtraction):

$$x = \frac{\tilde{x}}{\lVert \tilde{x} \rVert + \varepsilon},\qquad \varepsilon = 10^{-8}$$

**3. Layer forward pass** (`layer_loss_and_grads`, `FFNetwork.layer_outputs`):

$$z = Wx + b,\qquad y = \max(z, 0),\qquad G = \sum_j y_j^2,\qquad p = \sigma(G - \theta)$$

$G$ is the goodness; $p$ is the layer's probability that the input is positive.

**4. Local loss** (one per layer, batch mean):

$$L = \frac{1}{B}\sum_{i=1}^{B}\Big[\operatorname{softplus}\big(-(G^{\text{pos}}_i - \theta)\big) + \operatorname{softplus}\big(G^{\text{neg}}_i - \theta\big)\Big]$$

**5. Hand-derived gradient.** Using $\frac{d}{du}\operatorname{softplus}(u) = \sigma(u)$:

$$\frac{\partial L}{\partial G^{\text{pos}}_i} = -\frac{1 - p^{\text{pos}}_i}{B},\qquad
\frac{\partial L}{\partial G^{\text{neg}}_i} = \frac{p^{\text{neg}}_i}{B},\qquad
\frac{\partial G_i}{\partial z_{ji}} = 2\,y_{ji}\ \ (\text{zero where } z_{ji} \le 0)$$

$$\delta_i = \frac{\partial L}{\partial G_i}\, 2\,y_i,\qquad
\nabla W = \sum_i \big(\delta^{\text{pos}}_i (x^{\text{pos}}_i)^T + \delta^{\text{neg}}_i (x^{\text{neg}}_i)^T\big),\qquad
\nabla b = \sum_i \big(\delta^{\text{pos}}_i + \delta^{\text{neg}}_i\big)$$

In matrix form (examples as columns): $\nabla W = \Delta^{\text{pos}} X_{\text{pos}}^T + \Delta^{\text{neg}} X_{\text{neg}}^T$.
The update uses only the layer's own input $x$ and output $y$: a local outer product, no $W^T$, no backward pass.

**6. Passing to the next layer.** Only the direction of the activity is passed on (its length is this layer's goodness):

$$x_{\text{next}} = \frac{y}{\lVert y \rVert + \varepsilon}$$

**7. Initialization** (`FFNetwork.__init__`). With $\lVert x \rVert = 1$ and $W_{jk} \sim \mathcal{N}(0, s^2)$:
$\operatorname{Var}(z_j) = s^2$, $\mathbb{E}[\max(z_j,0)^2] = s^2/2$, so choosing $s^2 = 2/n_\text{out}$ gives
$\mathbb{E}[G] = n_\text{out}\, s^2/2 = 1$ for any width. Biases start at 0.

**8. Negative labels.**
- Random (`random_wrong_labels`): $k = (y + r) \bmod 10$ with $r$ uniform in $\lbrace 1,\dots,9 \rbrace$.
- Hard (`hard_wrong_labels`): with $G_k$ the total goodness of the image under label $k$ (from the previous epoch),

$$P(k) = \frac{e^{G_k}}{\sum_{j \ne y} e^{G_j}}\ \ \text{for } k \ne y,\qquad P(y) = 0$$

  Epoch 1 uses random labels (no goodness table yet).

**9. Optimizer** (`_Updater`), for each parameter $w$ with gradient $g$ at step $t$:
- Adam ($\eta = 0.003$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$):

$$m \leftarrow \beta_1 m + (1-\beta_1) g,\quad v \leftarrow \beta_2 v + (1-\beta_2) g^2,\quad
w \leftarrow w - \eta\,\frac{m/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)} + \epsilon}$$

- SGD with momentum $\mu$: $\ v \leftarrow \mu v - \eta g,\ \ w \leftarrow w + v$.

**10. Training loop** (`train_ff`):

```
for each epoch:
    wrong labels: random (epoch 1 or negatives="random") or hard (from last epoch's goodness table)
    for each shuffled batch of B images:
        x_pos = normalize(embed(images, true labels)); x_neg = normalize(embed(images, wrong labels))
        for each layer l:                         # all layers on every batch
            loss_l, grad_W, grad_b, y_pos, y_neg = layer step on (x_pos, x_neg)     # steps 3-5
            update W_l, b_l with the optimizer    # uses only layer l's own loss
            x_pos, x_neg = normalize(y_pos), normalize(y_neg)                       # step 6
    goodness table G[n, k] for all training images and labels -> train accuracy and next hard negatives
```

**11. Prediction** (`goodness_per_label`, `predict`): 10 forward passes per image,

$$\hat{y} = \arg\max_{k \in \lbrace 0,\dots,9 \rbrace} \sum_{l} G_l\big(\text{image with label } k\big)$$

(`skip_first=True` leaves layer 1 out of the sum; for 784-500-500 including it was better.)

**12. Gradient check** (`check_layer_gradients`):

$$g_{\text{fd}} = \frac{L(w+\epsilon) - L(w-\epsilon)}{2\epsilon},\quad \epsilon = 10^{-6},\qquad
\text{error} = \frac{|g - g_{\text{fd}}|}{\max(|g| + |g_{\text{fd}}|,\ 10^{-6})}$$

The $10^{-6}$ floor: in float64 the finite difference resolves gradients only to about $10^{-16}/\epsilon = 10^{-10}$.

## Backprop baseline (`src/bp_relu.py`)

**Forward** (`ReLUNetwork.forward`), layers $l = 1..L$, $a^{(0)} = x$ (pixels, not normalized):

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)},\qquad a^{(l)} = \max(z^{(l)}, 0)\ \ (l < L),\qquad a^{(L)} = \sigma(z^{(L)})$$

**Loss**: binary cross-entropy over the 10 outputs, one-hot target $t$:
$\ L = -\frac{1}{B}\sum_i \sum_k \big[t_k \log a_k + (1-t_k)\log(1-a_k)\big]$.

**Backward** (`gradients`), with $\delta^{(l)} = \partial L / \partial z^{(l)}$:

$$\delta^{(L)} = a^{(L)} - t,\qquad
\delta^{(l-1)} = \big(W^{(l)}\big)^T \delta^{(l)} \odot \mathbf{1}\big[z^{(l-1)} > 0\big],\qquad
\nabla W^{(l)} = \tfrac{1}{B}\,\delta^{(l)} \big(a^{(l-1)}\big)^T,\qquad
\nabla b^{(l)} = \tfrac{1}{B}\textstyle\sum_i \delta^{(l)}_i$$

$\delta^{(L)} = a - t$ because the sigmoid derivative $a(1-a)$ cancels with the cross-entropy derivative. The
$W^T$ step is the weight transport that FF avoids.

**Update** (`backprop_relu`): $W \leftarrow W - \eta \nabla W$, $\eta = 0.1$, batch 32.
**Initialization**: He, $W^{(l)}_{jk} \sim \mathcal{N}(0, 2/n_\text{in})$, biases 0.

## Crossbar mapping (`src/crossbar.py`, used by `ReLUCrossbarNetwork`)

For each layer, with the bias as an extra row driven at $V_\text{read}$:

$$W_\text{aug} = \begin{bmatrix} W^T \\ b^T \end{bmatrix},\qquad g = \frac{G_\text{max} - G_\text{min}}{\max |W_\text{aug}|},\qquad
G^{\pm} = G_\text{min} + g \max(\pm W_\text{aug}, 0)$$

$$G \leftarrow \operatorname{clip}\big(G + \mathcal{N}(0, \sigma^2),\ G_\text{min},\ G_\text{max}\big),\qquad
I_j = \sum_i G_{ij} V_i,\qquad
z = \frac{I^{+} - I^{-}}{V_\text{read}\, g},\quad V = V_\text{read}\,[a,\ 1]$$

$G_\text{min} = 0.1\ \mu S$, $G_\text{max} = 150\ \mu S$, $\sigma = 0.1\ \mu S$, $V_\text{read} = 0.2$ V. With $\sigma = 0$ the
crossbar output equals $Wa + b$ (checked to $2\times10^{-15}$). In FF every layer input has unit length, so every
row voltage is at most $V_\text{read}$.

## Usage

```python
from src.mnist import load_mnist
from src.ff import FFNetwork, train_ff

d = load_mnist()
net, hist = train_ff(FFNetwork([784, 500, 500], theta=5.0), *d["train"], epochs=60,
                     negatives="hard", X_test=d["val"][0], y_test=d["val"][1])
print((net.predict(d["test"][0]) == d["test"][1]).mean())
```

`ff.ipynb`: with `RETRAIN = False`, saved models in `../results/` are loaded and missing ones are trained (~15 min) and saved.

## Notes

- `history["test_acc"]` is accuracy on whatever is passed as `X_test` (the validation set here).
- Train `ReLUNetwork` with `backprop_relu`; `backprop.backprop()` uses one activation derivative for all layers and raises an error for this network.
- θ = 5 rather than θ = layer width: layer inputs have unit length, so G starts near 1. θ = 1 / 2 / 5 gave 95.9 / 96.5 / 97.4 % val at 10 epochs.
- Adam rather than SGD: on 10k images / 5 epochs, SGD + momentum reached 88.0 % val at best (lr 0.3) vs 92.2 % for Adam, and lr ≥ 1 killed all ReLUs. SGD remains available (`optimizer="sgd"`) for in-situ crossbar training.
- Gradient checks: `check_layer_gradients` (FF), `check_gradients` (BP).
- macOS: if `load_mnist()` fails with `CERTIFICATE_VERIFY_FAILED`, run `Install Certificates.command` from the Python folder in Applications.

## Open items

- FF on the crossbar: ex-situ transfer, then in-situ if required.
- Noise sweep σ = 0–2 µS; θ = 10–20; early stopping (train accuracy reaches 100 % by ~epoch 30).

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

## Method

- Label written into pixels 0–9 (always 0 in MNIST). Positive = image + true label; negative = image + wrong label.
- Each layer: z = Wx + b, y = ReLU(z), goodness G = Σy². Loss: softplus(θ − G_pos) + softplus(G_neg − θ).
- Update ∇W = δxᵀ with δ = (∂L/∂G)·2y: local outer product, no backward pass, no Wᵀ.
- Layer input normalized to unit length; only the direction is passed on.
- Prediction: embed each label, sum goodness over layers, argmax.

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

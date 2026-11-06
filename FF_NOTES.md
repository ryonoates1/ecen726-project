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

| Model | Hidden | Val | Test |
|---|---|---|---|
| BP, sigmoid | 64-64 | 97.13 % | 97.52 % |
| BP, ReLU | 64-64 | 97.49 % | 97.78 % |
| BP, ReLU | 500-500 | 98.22 % | 98.40 % |
| FF | 500-500 | 97.90 % | 97.91 % |

FF: θ = 5, Adam lr 0.003, batch 100, hard negatives, 60 epochs. Hyperparameters chosen on the validation set.

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
- Adam rather than SGD: SGD with lr ≥ 1 killed all ReLUs; smaller lr trained slowly. SGD remains available (`optimizer="sgd"`).
- Gradient checks: `check_layer_gradients` (FF), `check_gradients` (BP).
- macOS: if `load_mnist()` fails with `CERTIFICATE_VERIFY_FAILED`, run `Install Certificates.command` from the Python folder in Applications.

## Open items

- FF on the crossbar: ex-situ transfer, then in-situ if required.
- Noise sweep σ = 0–2 µS; θ = 10–20; early stopping (train accuracy reaches 100 % by ~epoch 30).

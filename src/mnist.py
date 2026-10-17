"""Load MNIST, split it into train/val/test sets, and plot individual examples."""

import gzip
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np

# Mirror of the original Yann LeCun files (same one torchvision uses).
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mnist"


def _download(data_dir=DATA_DIR):
    """Download any MNIST files that aren't already cached in data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in MNIST_FILES.values():
        path = data_dir / filename
        if not path.exists():
            print(f"Downloading {filename}...")
            urlretrieve(MNIST_URL + filename, path)


def _read_images(path):
    """Parse an IDX image file into a (N, 28, 28) uint8 array."""
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def _read_labels(path):
    """Parse an IDX label file into a (N,) uint8 array."""
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)


def load_mnist(val_size=10_000, normalize=True, seed=0, data_dir=DATA_DIR):
    """Load MNIST and split it into training, validation, and test sets.

    The official 60k training set is shuffled and split into train/val;
    the official 10k test set is kept as-is.

    Args:
        val_size: Number of training examples held out for validation.
        normalize: If True, scale pixels to float32 in [0, 1]; otherwise
            leave them as uint8 in [0, 255].
        seed: Random seed for the train/val shuffle.
        data_dir: Where the raw .gz files are cached.

    Returns:
        Dict with keys "train", "val", "test", each mapping to (X, y) where
        X has shape (N, 28, 28) and y has shape (N,).
    """
    _download(data_dir)
    X_train = _read_images(data_dir / MNIST_FILES["train_images"])
    y_train = _read_labels(data_dir / MNIST_FILES["train_labels"])
    X_test = _read_images(data_dir / MNIST_FILES["test_images"])
    y_test = _read_labels(data_dir / MNIST_FILES["test_labels"])

    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_train))
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    return {
        "train": (X_train[train_idx], y_train[train_idx]),
        "val": (X_train[val_idx], y_train[val_idx]),
        "test": (X_test, y_test),
    }


def plot_example(X, y, i, ax=None):
    """Plot the ith image in X with its label from y."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(X[i])
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(f"Example {i}: label = {y[i]}")
    ax.axis("off")
    return ax


if __name__ == "__main__":
    splits = load_mnist()
    for name, (X, y) in splits.items():
        print(f"{name:>5}: X {X.shape}, y {y.shape}")

    X_train, y_train = splits["train"]
    plot_example(X_train, y_train, 0)
    plt.show()

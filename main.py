import typer
import numpy as np
import idx2numpy
from tqdm import tqdm
import json
import random

np.random.seed(42)
random.seed(420)


def main(viz: bool = False):

    X_train = idx2numpy.convert_from_file("data/mnist/train-images-idx3-ubyte")
    y_train = idx2numpy.convert_from_file("data/mnist/train-labels-idx1-ubyte")

    X_test = idx2numpy.convert_from_file("data/mnist/t10k-images-idx3-ubyte")
    y_test = idx2numpy.convert_from_file("data/mnist/t10k-labels-idx1-ubyte")

    X_train = to_sparse(X_train)
    X_test = to_sparse(X_test)

    max_length = max(x.shape[0] for x in X_train + X_test)

    np.save(f"data/point-cloud-mnist-2D/y_train.npy", y_train)
    np.save(f"data/point-cloud-mnist-2D/y_test.npy", y_test)

    for length in [50, 100, 200, max_length]:
        X_train_sample = sample(X_train, length, "train")
        X_test_sample = sample(X_test, length, "test")

        suffix = "max" if length == max_length else length

        np.save(f"data/point-cloud-mnist-2D/X_train_{suffix}.npy", X_train_sample)
        np.save(f"data/point-cloud-mnist-2D/X_test_{suffix}.npy", X_test_sample)


def sample(X, k, name):

    samples = []

    padding = np.array([[-1, -1, -1]], dtype=np.int32)

    for x in tqdm(X, desc=f"Sampling {name}_{k}"):
        N = len(x)
        p = x[:, 2] / x[:, 2].sum()

        if N > k:
            idx = np.random.choice(N, k, p=p, replace=False)
            x = x[idx]
        elif N < k:
            x = np.concatenate([x, np.tile(padding, (k - N, 1))])

        samples.append(x)

    samples = np.stack(samples, axis=0).astype(np.int32)

    return samples


def to_sparse(X):

    N = len(X)
    w, h = X.shape[1:]

    xx, yy = np.meshgrid(np.arange(w), np.arange(h)[::-1])

    xx = xx[None]

    xx = np.tile(xx, [N, 1, 1])

    yy = yy[None]
    yy = np.tile(yy, [N, 1, 1])

    X = np.stack([xx, yy, X], axis=-1)
    X = X.reshape(N, -1, 3)

    return [Xi[Xi[:, 2] > 0] for Xi in X]


if __name__ == "__main__":
    typer.run(main)

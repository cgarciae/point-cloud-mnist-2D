import json
import os
import random

import idx2numpy
import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

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

    X_train = pad(X_train, max_length, "train")
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = pad(X_test, max_length, "test")
    X_test = X_test.reshape(X_test.shape[0], -1)

    data_train = np.concatenate([y_train[:, None], X_train], axis=-1)
    data_test = np.concatenate([y_test[:, None], X_test], axis=-1)

    columns = [[f"x{i}", f"y{i}", f"v{i}"] for i in range(max_length)]
    columns = ["label"] + sum(columns, [])

    df_train = pd.DataFrame(data_train, columns=columns)
    df_test = pd.DataFrame(data_test[:, 1:], columns=columns[1:])
    test_labels = pd.DataFrame(data_test[:, :1], columns=columns[:1])

    os.makedirs("data/point-cloud-mnist-2D", exist_ok=True)

    print("saving train")
    df_train.to_csv("data/point-cloud-mnist-2D/train.csv", index=False)
    print("saving test")
    df_test.to_csv("data/point-cloud-mnist-2D/test.csv", index=False)
    print("saving test labels")
    test_labels.to_csv("data/point-cloud-mnist-2D/test_labels.csv", index=False)


def pad(X, k, name):

    samples = []

    padding = np.array([[-1, -1, -1]], dtype=np.int32)

    for x in tqdm(X, desc=f"Padding {name}_{k}"):
        N = len(x)

        if N < k:
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

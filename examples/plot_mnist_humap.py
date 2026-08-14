"""Display a HUMAP embedding for MNIST.

Run from the project root after installing HUMAP:

    python examples/plot_mnist_humap.py

The example uses a subset by default so it remains practical for a quick demo.
"""

import argparse

import humap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Display a HUMAP MNIST embedding.")
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of MNIST samples to use. Use 70000 for the full dataset.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=2,
        help="Hierarchy level to display.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=30,
        help="Number of neighbors for HUMAP.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the figure, for example figs/mnist_humap_level2.png.",
    )
    return parser.parse_args()


def load_mnist(samples):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=True)

    if samples and samples < len(X):
        _, X, _, y = train_test_split(
            X,
            y,
            test_size=samples,
            random_state=0,
            stratify=y,
        )

    X = X.to_numpy(dtype=np.float32) / 255.0
    y = y.to_numpy(dtype=int)
    return X, y


def main():
    args = parse_args()
    X, y = load_mnist(args.samples)

    reducer = humap.HUMAP(
        np.array([0.2, 0.2]),
        n_neighbors=args.neighbors,
        init="Random",
        reproducible=True,
        verbose=True,
    )
    reducer.fit(X, y)

    embedding = reducer.transform(args.level)
    labels = y if args.level == 0 else reducer.labels(args.level)

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.85,
        linewidths=0,
    )

    ax.set_title(f"HUMAP MNIST - level {args.level}")
    ax.set_xlabel("HUMAP 1")
    ax.set_ylabel("HUMAP 2")
    ax.set_aspect("equal", adjustable="datalim")

    cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(10))
    cbar.set_label("Digit")

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=200)

    plt.show()


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import humap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit HUMAP on MNIST and save a level embedding plot."
    )
    parser.add_argument(
        "--level",
        type=int,
        default=2,
        help="Hierarchy level to embed and visualize.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of MNIST samples to use. Use 0 for the full dataset.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Number of neighbors used by HUMAP.",
    )
    parser.add_argument(
        "--output",
        default="humap_mnist_level2.png",
        help="Output image file for the plot.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window after saving.",
    )
    return parser.parse_args()


def load_mnist(sample_size, random_state):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=True)
    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy().astype(int)

    if sample_size and sample_size < X.shape[0]:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            stratify=y,
            random_state=random_state,
        )

    return normalize(X), y


def main():
    args = parse_args()

    print("Loading MNIST...")
    X, y = load_mnist(args.sample_size, args.random_state)

    print(f"Fitting HUMAP on {X.shape[0]} samples...")
    h_umap = humap.HUMAP(
        levels=np.array([0.2, 0.2]),
        n_neighbors=args.n_neighbors,
        verbose=True,
        reproducible=True,
    )
    h_umap.fit(X, y)

    print(f"Embedding level {args.level}...")
    embedding = h_umap.transform(args.level)
    labels = h_umap.labels(args.level) if args.level > 0 else y

    if embedding.shape[1] < 2:
        raise ValueError("The embedding must have at least two dimensions to plot.")

    output = Path(args.output).resolve()
    fig, ax = plt.subplots(figsize=(11, 8))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels.astype(int),
        cmap="tab10",
        s=8,
        alpha=0.82,
        linewidths=0,
    )
    ax.set_title(f"HUMAP MNIST - level {args.level}")
    ax.set_xlabel("HUMAP 1")
    ax.set_ylabel("HUMAP 2")
    ax.grid(True, alpha=0.18)

    colorbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(10))
    colorbar.set_label("Digit")

    fig.tight_layout()
    fig.savefig(output, dpi=180)

    print(f"Saved plot to: {output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

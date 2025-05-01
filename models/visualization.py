import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_embeddings(
    embeddings,
    labels,
    method="tsne",
    title="Embedding Visualization",
    save_path=None,
    perturbed_embeddings=None,
):
    plt.figure(figsize=(12, 10))

    if perturbed_embeddings is None:
        if method == "pca":
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)

        reduced_embeddings = reducer.fit_transform(embeddings)
        df = pd.DataFrame(
            {
                "x": reduced_embeddings[:, 0],
                "y": reduced_embeddings[:, 1],
                "label": ["Real Art" if l == 1 else "AI-Generated Art" for l in labels],
            }
        )

        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="label",
            palette={"Real Art": "blue", "AI-Generated Art": "red"},
            alpha=0.7,
            s=100,
        )

    else:
        combined_embeddings = np.vstack([embeddings, perturbed_embeddings])
        combined_labels = np.concatenate(
            [labels, np.zeros(perturbed_embeddings.shape[0])]
        )
        categories = np.concatenate(
            [
                np.array(["Original"] * len(labels)),
                np.array(["Perturbed"] * len(perturbed_embeddings)),
            ]
        )

        if method == "pca":
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)

        reduced_embeddings = reducer.fit_transform(combined_embeddings)
        df = pd.DataFrame(
            {
                "x": reduced_embeddings[:, 0],
                "y": reduced_embeddings[:, 1],
                "label": [
                    "Real Art" if l == 1 else "AI-Generated Art"
                    for l in combined_labels
                ],
                "category": categories,
            }
        )

        palette = {
            ("Real Art", "Original"): "blue",
            ("AI-Generated Art", "Original"): "red",
            ("AI-Generated Art", "Perturbed"): "green",
        }

        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="label",
            style="category",
            palette={"Real Art": "blue", "AI-Generated Art": "red"},
            alpha=0.7,
            s=100,
        )

        perturbed_df = df[df["category"] == "Perturbed"]
        plt.scatter(
            perturbed_df["x"],
            perturbed_df["y"],
            color="green",
            marker="x",
            s=150,
            alpha=0.8,
            label="Perturbed AI Art",
        )

    plt.title(title, fontsize=16)
    plt.xlabel(f"{method.upper()} Component 1", fontsize=14)
    plt.ylabel(f"{method.upper()} Component 2", fontsize=14)
    plt.legend(fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")

    plt.show()

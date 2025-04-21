from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_embeddings(embeddings, labels, method='tsne', title='Embedding Visualization',
                         save_path=None, perturbed_embeddings=None):
    """
    Visualize high-dimensional embeddings in 2D using PCA or t-SNE.

    Parameters:
        embeddings: numpy array of embeddings with shape (n_samples, n_features)
        labels: numpy array of labels (0 for AI-generated, 1 for real)
        method: 'pca' or 'tsne'
        title: title for the plot
        save_path: path to save the visualization
        perturbed_embeddings: optional, perturbed AI embeddings to visualize
    """
    plt.figure(figsize=(12, 10))

    # Create a DataFrame for easier plotting
    if perturbed_embeddings is None:
        # Standard visualization for original embeddings
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:  # t-SNE
            reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)

        reduced_embeddings = reducer.fit_transform(embeddings)

        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'label': ['Real Art' if l == 1 else 'AI-Generated Art' for l in labels]
        })

        # Create the scatter plot
        sns.scatterplot(data=df, x='x', y='y', hue='label', palette={'Real Art': 'blue', 'AI-Generated Art': 'red'},
                        alpha=0.7, s=100)

    else:
        # Visualization with perturbed embeddings
        # Combine original and perturbed embeddings
        combined_embeddings = np.vstack([embeddings, perturbed_embeddings])

        # Create labels for the combined dataset
        combined_labels = np.concatenate([
            labels,  # Original labels
            np.zeros(perturbed_embeddings.shape[0])  # Perturbed embeddings (all AI-generated)
        ])

        # Create a category column
        categories = np.concatenate([
            np.array(['Original'] * len(labels)),
            np.array(['Perturbed'] * len(perturbed_embeddings))
        ])

        # Reduce dimensions
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:  # t-SNE
            reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)

        reduced_embeddings = reducer.fit_transform(combined_embeddings)

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'label': ['Real Art' if l == 1 else 'AI-Generated Art' for l in combined_labels],
            'category': categories
        })

        # Create a custom color palette
        palette = {
            ('Real Art', 'Original'): 'blue',
            ('AI-Generated Art', 'Original'): 'red',
            ('AI-Generated Art', 'Perturbed'): 'green'
        }

        # Create the scatter plot
        sns.scatterplot(
            data=df, x='x', y='y',
            hue='label', style='category',
            palette={'Real Art': 'blue', 'AI-Generated Art': 'red'},
            alpha=0.7, s=100
        )

        # Add a separate scatter for perturbed embeddings
        perturbed_df = df[df['category'] == 'Perturbed']
        plt.scatter(perturbed_df['x'], perturbed_df['y'], color='green', marker='x', s=150, alpha=0.8, label='Perturbed AI Art')

    plt.title(title, fontsize=16)
    plt.xlabel(f"{method.upper()} Component 1", fontsize=14)
    plt.ylabel(f"{method.upper()} Component 2", fontsize=14)
    plt.legend(fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    # plt.show()

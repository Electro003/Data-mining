import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import DBSCAN
from minisom import MiniSom
import os

def normalize_data(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def apply_kmeans(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels, kmeans


def apply_gmm(features, num_clusters):
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    labels = gmm.fit_predict(features)
    return labels, gmm


def apply_hierarchical(features, num_clusters, linkage='ward'):
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage)
    labels = hierarchical.fit_predict(features)
    return labels


def apply_dbscan(features, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)
    return labels


def apply_som(features, map_size=(10, 10), sigma=1.0, learning_rate=0.5, iterations=1000, random_seed=42):
    n_features = features.shape[1]
    som = MiniSom(
        map_size[0],
        map_size[1],
        n_features,
        sigma=sigma,
        learning_rate=learning_rate,
        neighborhood_function='gaussian',
        random_seed=random_seed
    )

    som.random_weights_init(features)
    som.train(features, iterations)

    bmu_indices = np.array([som.winner(x) for x in features])

    weights = som.get_weights()

    kmeans = KMeans(n_clusters=3, random_state=random_seed)
    kmeans.fit(weights.reshape(-1, n_features))

    cluster_labels = np.zeros(len(features), dtype=int)
    for i, bmu in enumerate(bmu_indices):
        cluster_labels[i] = kmeans.labels_[bmu[0] * map_size[1] + bmu[1]]

    return cluster_labels, som



def compute_ari(true_labels, predicted_labels):
    return adjusted_rand_score(true_labels, predicted_labels)


# def apply_pca_and_plot(features, true_labels, predicted_labels, plot_title, file_name):
#     pca = PCA(n_components=2)
#     reduced_features = pca.fit_transform(features)
#
#     # Predefined colors for clusters (ensure sufficient colors for clusters)
#     cluster_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'pink', 'gray', 'cyan']
#
#     # Get unique labels from both true and predicted labels
#     unique_labels = np.unique(np.concatenate([true_labels, predicted_labels]))  # Combine both true and predicted labels
#     color_map = {label: cluster_colors[i % len(cluster_colors)] for i, label in enumerate(unique_labels)}
#
#     # Map the true and predicted labels to their corresponding colors
#     true_colors = np.array([color_map[label] for label in true_labels])
#     pred_colors = np.array([color_map[label] for label in predicted_labels])
#
#     # Create the figure for side-by-side plots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
#
#     # Plot for True Labels (Left Plot)
#     scatter_true = ax1.scatter(reduced_features[:, 0], reduced_features[:, 1], c=true_colors, marker='o', alpha=0.7)
#     ax1.set_title(f'{plot_title} - True Labels')
#     ax1.set_xlabel('PCA Component 1')
#     ax1.set_ylabel('PCA Component 2')
#
#     # Plot for Predicted Labels (Right Plot)
#     scatter_pred = ax2.scatter(reduced_features[:, 0], reduced_features[:, 1], c=pred_colors, marker='x', alpha=0.7)
#     ax2.set_title(f'{plot_title} - Predicted Labels')
#     ax2.set_xlabel('PCA Component 1')
#     ax2.set_ylabel('PCA Component 2')
#
#     legend_labels = [f'Cluster {i + 1}' for i in range(len(unique_labels))]
#     handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label], markersize=10)
#                for label in unique_labels]
#
#     fig.legend(handles, legend_labels, loc='upper center', ncol=len(unique_labels), bbox_to_anchor=(0.5, 1.05))
#
#     plt.tight_layout()
#     plt.savefig(f'task1_visualization/{file_name}_pca_plot_side_by_side.png')
#     plt.close()




def apply_pca_and_plot(features, true_labels, predicted_labels, title, output_filename=None):

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)

    ari = adjusted_rand_score(true_labels, predicted_labels)

    all_labels = np.unique(np.concatenate([true_labels, predicted_labels]))

    cmap = plt.cm.get_cmap('tab10', len(all_labels))
    color_dict = {label: cmap(i) for i, label in enumerate(all_labels)}


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for label in np.unique(true_labels):
        mask = true_labels == label
        ax1.scatter(
            pca_result[mask, 0],
            pca_result[mask, 1],
            c=[color_dict[label]],
            label=f'Cluster {label}',
            alpha=0.7,
            edgecolors='w',
            s=70
        )

    ax1.set_title('True Labels', fontsize=14)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    for label in np.unique(predicted_labels):
        mask = predicted_labels == label
        ax2.scatter(
            pca_result[mask, 0],
            pca_result[mask, 1],
            c=[color_dict[label]],
            label=f'Cluster {label}',
            alpha=0.7,
            edgecolors='w',
            s=70
        )

    ax2.set_title('Predicted Labels', fontsize=14)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.03),
        ncol=min(5, len(handles)),
        fontsize=12
    )

    fig.text(
        0.5, 0.96,
        f'{title} (Adjusted Rand Index: {ari:.4f})',
        horizontalalignment='center',
        fontsize=16,
        weight='bold'
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])


    if output_filename:
        os.makedirs(os.path.dirname(output_filename) if os.path.dirname(output_filename) else '.', exist_ok=True)
        plt.savefig(f"task1_visualization/{output_filename}.png", dpi=300, bbox_inches='tight')



    return pca, pca_result, ari

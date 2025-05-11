import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from matplotlib import cm


def visualize_clusters_pca(features, labels, n_clusters, algorithm, dataset_name, output_dir, title_suffix=''):
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Create colormap
    cmap = cm.get_cmap('tab10', n_clusters)

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot each cluster with a different color
    for i in range(n_clusters):
        mask = labels == i
        plt.scatter(
            reduced_features[mask, 0],
            reduced_features[mask, 1],
            c=[cmap(i)],
            label=f'Cluster {i}',
            alpha=0.7,
            edgecolors='w',
            s=70
        )

    # Add explained variance as axis labels
    explained_var_ratio = pca.explained_variance_ratio_
    plt.xlabel(f'PC1 ({explained_var_ratio[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({explained_var_ratio[1]:.2%} variance)')

    plt.title(f'{algorithm} Clustering ({n_clusters} clusters) - {dataset_name}{title_suffix}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot
    plot_filename = os.path.join(
        output_dir,
        f"{dataset_name}_{algorithm}_{n_clusters}clusters_pca.png"
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"PCA visualization saved to {plot_filename}")


def visualize_optimal_clustering_results(features, method_results, dataset_name, output_dir):
    for method, optimal_k in method_results.items():
        if 'KMeans' in method:
            algorithm = 'KMeans'
            clustering = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        elif 'Hierarchical' in method:
            algorithm = 'Hierarchical'
            clustering = AgglomerativeClustering(n_clusters=optimal_k)
        else:
            continue

        # Apply clustering
        labels = clustering.fit_predict(features)

        # Visualize with PCA
        visualize_clusters_pca(
            features,
            labels,
            optimal_k,
            algorithm,
            dataset_name,
            output_dir,
            title_suffix=f' (Method: {method.split("_")[1]})'
        )


def plot_silhouette_visualization(features, optimal_k, algorithm, dataset_name, output_dir):
    from sklearn.metrics import silhouette_samples

    if algorithm == 'kmeans':
        clustering = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    elif algorithm == 'hierarchical':
        clustering = AgglomerativeClustering(n_clusters=optimal_k)
    else:
        raise ValueError("Algorithm must be 'kmeans' or 'hierarchical'")

    cluster_labels = clustering.fit_predict(features)

    sample_silhouette_values = silhouette_samples(features, cluster_labels)

    plt.figure(figsize=(12, 8))

    y_lower = 10

    for i in range(optimal_k):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        cmap = cm.get_cmap('tab10', optimal_k)
        color = cmap(i)

        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}')

        y_lower = y_upper + 10

    avg_silhouette = np.mean(sample_silhouette_values)

    plt.axvline(x=avg_silhouette, color='red', linestyle='--',
                label=f'Average Silhouette: {avg_silhouette:.3f}')

    plt.title(f'Silhouette Analysis ({algorithm.capitalize()}, K={optimal_k}) - {dataset_name}')
    plt.xlabel('Silhouette Coefficient Values')
    plt.ylabel('Cluster Label')

    plt.xlim([-0.1, 1])

    plt.ylim([0, y_lower + 10])

    plt.legend(loc='best')
    plt.tight_layout()

    plot_filename = os.path.join(
        output_dir,
        f"{dataset_name}_{algorithm}_silhouette_analysis_k{optimal_k}.png"
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

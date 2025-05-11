# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.metrics import silhouette_score
# from scipy.cluster.hierarchy import dendrogram, linkage
# from kneed import KneeLocator
#
#
# def perform_kmeans_elbow_analysis(features, cluster_range, dataset_name, output_dir):
#     inertias = []
#
#     for k in cluster_range:
#         kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#         kmeans.fit(features)
#         inertias.append(kmeans.inertia_)
#
#     # Use KneeLocator to automatically find the elbow point
#     try:
#         kneedle = KneeLocator(
#             list(cluster_range),
#             inertias,
#             curve='convex',
#             direction='decreasing',
#             online=True
#         )
#         optimal_k = kneedle.elbow
#     except:
#         # If knee cannot be detected, use a fallback method
#         # Calculate the differences between consecutive inertia values
#         # Calculate the rate of change of the differences
#         # The elbow is where the rate of change is maximum
#         print("-----AUTOMATIC ELBOW DETECTION FAILED!----")
#         # This shift by 2 is arbitrary and doesn't have a clear mathematical justification
#         diffs = np.diff(inertias)
#         rate_of_change = np.diff(diffs)
#         optimal_k = cluster_range[np.argmax(np.abs(rate_of_change)) + 2]
#
#     # If still no optimal k, just pick the middle of the range
#     if optimal_k is None:
#         print("-----ELBOW ALGORITHM DID NOT WORK!. MIDDLE OF CLUSTER RANGE----")
#         optimal_k = cluster_range[len(cluster_range) // 2]
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(cluster_range, inertias, 'bo-')
#     plt.xlabel('Number of Clusters (K)')
#     plt.ylabel('Inertia')
#     plt.title(f'Elbow Method for Optimal K (KMeans) - {dataset_name}')
#
#     # Mark the optimal k
#     plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K = {optimal_k}')
#     plt.legend()
#     plt.grid(True)
#
#     # Save the plot
#     plot_filename = os.path.join(output_dir, f"{dataset_name}_kmeans_elbow.png")
#     plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
#     plt.close()
#
#     print(f"K-means elbow analysis optimal K: {optimal_k}")
#     return inertias, optimal_k
#
#
# def perform_hierarchical_elbow_analysis(features, cluster_range, dataset_name, output_dir):
#     Z = linkage(features, method='ward')
#
#     plt.figure(figsize=(12, 8))
#     dendrogram(
#         Z,
#         truncate_mode='none',
#         p=5,  # this is ignored
#         leaf_font_size=10,
#     )
#     plt.title(f'Hierarchical Clustering Dendrogram - {dataset_name}')
#     plt.xlabel('Sample index or Cluster size')
#     plt.ylabel('Distance')
#
#     dendrogram_filename = os.path.join(output_dir, f"{dataset_name}_hierarchical_dendrogram.png")
#     plt.savefig(dendrogram_filename, dpi=300, bbox_inches='tight')
#     plt.close()
#
#     # Extract the distances for the last n_samples-1 merges
#     last_n_distances = Z[:, 2]
#
#     # Determine the number of clusters to extract from the linkage matrix
#     # We'll use the last len(cluster_range) merges (i.e., going from max(cluster_range) down to 1 cluster)
#     max_k = max(cluster_range)
#     relevant_distances = last_n_distances[-max_k + 1:]
#
#     # Reverse the distances to match number of clusters (from 2 to max_k)
#     relevant_distances = relevant_distances[::-1]
#
#     # For plotting, we might need to extend the distances list
#     distances = list(relevant_distances) + [0] * (len(cluster_range) - len(relevant_distances))
#     distances = distances[:len(cluster_range)]
#
#     # Use KneeLocator to automatically find the elbow point
#     try:
#         kneedle = KneeLocator(
#             list(cluster_range),
#             distances,
#             curve='convex',
#             direction='decreasing',
#             online=True
#         )
#         optimal_k = kneedle.elbow
#     except:
#         diffs = np.diff(distances)
#         rate_of_change = np.diff(diffs)
#         optimal_k = cluster_range[np.argmax(np.abs(rate_of_change)) + 2]
#
#     if optimal_k is None:
#         optimal_k = cluster_range[len(cluster_range) // 2]
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(cluster_range, distances, 'bo-')
#     plt.xlabel('Number of Clusters (K)')
#     plt.ylabel('Merge Distance')
#     plt.title(f'Elbow Method for Optimal K (Hierarchical) - {dataset_name}')
#
#     plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K = {optimal_k}')
#     plt.legend()
#     plt.grid(True)
#
#     # Save the plot
#     plot_filename = os.path.join(output_dir, f"{dataset_name}_hierarchical_elbow.png")
#     plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
#     plt.close()
#
#     print(f"Hierarchical elbow analysis optimal K: {optimal_k}")
#     return distances, optimal_k
#
#
# def perform_silhouette_analysis(features, cluster_range, algorithm, dataset_name, output_dir):
#     silhouette_scores = []
#
#     for k in cluster_range:
#         # Skip k=1 as silhouette score is not defined for a single cluster
#         if k <= 1:
#             silhouette_scores.append(0)
#             continue
#
#         # Apply clustering algorithm
#         if algorithm == 'kmeans':
#             clustering = KMeans(n_clusters=k, random_state=42, n_init=10)
#         elif algorithm == 'hierarchical':
#             clustering = AgglomerativeClustering(n_clusters=k)
#         else:
#             raise ValueError("Algorithm must be 'kmeans' or 'hierarchical'")
#
#         cluster_labels = clustering.fit_predict(features)
#
#         # Calculate silhouette score
#         # Handle the case where a cluster might have only one sample
#         try:
#             score = silhouette_score(features, cluster_labels)
#         except:
#             score = -1  # Invalid score if there's an error
#
#         silhouette_scores.append(score)
#
#     # Find the optimal k (highest silhouette score)
#     optimal_k = cluster_range[np.argmax(silhouette_scores)]
#
#     # Plot the silhouette scores
#     plt.figure(figsize=(10, 6))
#     plt.plot(cluster_range, silhouette_scores, 'bo-')
#     plt.xlabel('Number of Clusters (K)')
#     plt.ylabel('Silhouette Score')
#     plt.title(f'Silhouette Analysis for Optimal K ({algorithm.capitalize()}) - {dataset_name}')
#
#     plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K = {optimal_k}')
#     plt.legend()
#     plt.grid(True)
#
#
#     plot_filename = os.path.join(output_dir, f"{dataset_name}_{algorithm}_silhouette.png")
#     plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
#     plt.close()
#
#     print(f"{algorithm.capitalize()} silhouette analysis optimal K: {optimal_k}")
#     return silhouette_scores, optimal_k
#
#
# def calculate_cluster_metrics_for_k(features, k, algorithm='kmeans'):
#     if algorithm == 'kmeans':
#         clustering = KMeans(n_clusters=k, random_state=42, n_init=10)
#         cluster_labels = clustering.fit_predict(features)
#         inertia = clustering.inertia_
#     elif algorithm == 'hierarchical':
#         clustering = AgglomerativeClustering(n_clusters=k)
#         cluster_labels = clustering.fit_predict(features)
#         inertia = None
#     else:
#         raise ValueError("Algorithm must be 'kmeans' or 'hierarchical'")
#
#     silhouette = silhouette_score(features, cluster_labels) if k > 1 else 0
#
#     metrics = {
#         'k': k,
#         'algorithm': algorithm,
#         'inertia': inertia,
#         'silhouette': silhouette
#     }
#
#     return metrics


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from kneed import KneeLocator


def perform_kmeans_elbow_analysis(features, cluster_range, dataset_name, output_dir):
    inertias = []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    try:
        kneedle = KneeLocator(
            list(cluster_range),
            inertias,
            curve='convex',
            direction='decreasing',
            online=True
        )
        optimal_k = kneedle.elbow
    except:
        print("-----AUTOMATIC ELBOW DETECTION FAILED!----")
        diffs = np.diff(inertias)
        rate_of_change = np.diff(diffs)
        optimal_k = cluster_range[np.argmax(np.abs(rate_of_change)) + 2]

    if optimal_k is None:
        print("-----ELBOW ALGORITHM DID NOT WORK!. MIDDLE OF CLUSTER RANGE----")
        optimal_k = cluster_range[len(cluster_range) // 2]

    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title(f'Elbow Method for Optimal K (KMeans) - {dataset_name}')

    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K = {optimal_k}')
    plt.legend()
    plt.grid(True)

    plot_filename = os.path.join(output_dir, f"{dataset_name}_kmeans_elbow.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"K-means elbow analysis optimal K: {optimal_k}")
    return inertias, optimal_k


def perform_hierarchical_elbow_analysis(features, cluster_range, dataset_name, output_dir, linkage_method='ward'):
    Z = linkage(features, method=linkage_method)

    plt.figure(figsize=(12, 8))
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=15,
        leaf_font_size=10,
    )
    plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method}) - {dataset_name}')
    plt.xlabel('Sample index or Cluster size')
    plt.ylabel('Distance')

    dendrogram_filename = os.path.join(output_dir, f"{dataset_name}_hierarchical_dendrogram_{linkage_method}.png")
    plt.savefig(dendrogram_filename, dpi=300, bbox_inches='tight')
    plt.close()

    last_n_distances = Z[:, 2]

    max_k = max(cluster_range)
    relevant_distances = last_n_distances[-max_k + 1:]

    relevant_distances = relevant_distances[::-1]

    distances = list(relevant_distances) + [0] * (len(cluster_range) - len(relevant_distances))
    distances = distances[:len(cluster_range)]

    try:
        kneedle = KneeLocator(
            list(cluster_range),
            distances,
            curve='convex',
            direction='decreasing',
            online=True
        )
        optimal_k = kneedle.elbow
    except:
        diffs = np.diff(distances)
        rate_of_change = np.diff(diffs)
        optimal_k = cluster_range[np.argmax(np.abs(rate_of_change)) + 2]

    if optimal_k is None:
        optimal_k = cluster_range[len(cluster_range) // 2]

    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, distances, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Merge Distance')
    plt.title(f'Elbow Method for Optimal K (Hierarchical {linkage_method}) - {dataset_name}')

    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K = {optimal_k}')
    plt.legend()
    plt.grid(True)

    plot_filename = os.path.join(output_dir, f"{dataset_name}_hierarchical_elbow_{linkage_method}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Hierarchical elbow analysis ({linkage_method}) optimal K: {optimal_k}")
    return distances, optimal_k


def perform_silhouette_analysis(features, cluster_range, algorithm, dataset_name, output_dir, linkage_method='ward'):
    silhouette_scores = []

    for k in cluster_range:
        if k <= 1:
            silhouette_scores.append(0)
            continue

        if algorithm == 'kmeans':
            clustering = KMeans(n_clusters=k, random_state=42, n_init=10)
        elif algorithm == 'hierarchical':
            clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        else:
            raise ValueError("Algorithm must be 'kmeans' or 'hierarchical'")

        cluster_labels = clustering.fit_predict(features)

        try:
            score = silhouette_score(features, cluster_labels)
        except:
            score = -1

        silhouette_scores.append(score)

    optimal_k = cluster_range[np.argmax(silhouette_scores)]

    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')

    algorithm_title = algorithm
    if algorithm == 'hierarchical':
        algorithm_title = f'{algorithm} ({linkage_method})'

    plt.title(f'Silhouette Analysis for Optimal K ({algorithm_title.capitalize()}) - {dataset_name}')

    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K = {optimal_k}')
    plt.legend()
    plt.grid(True)

    filename_suffix = ''
    if algorithm == 'hierarchical':
        filename_suffix = f'_{linkage_method}'

    plot_filename = os.path.join(output_dir, f"{dataset_name}_{algorithm}{filename_suffix}_silhouette.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"{algorithm.capitalize()} silhouette analysis ({linkage_method if algorithm == 'hierarchical' else ''}) optimal K: {optimal_k}")
    return silhouette_scores, optimal_k


def plot_silhouette_visualization(features, optimal_k, algorithm, dataset_name, output_dir, linkage_method='ward'):
    from sklearn.metrics import silhouette_samples

    if algorithm == 'kmeans':
        clustering = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    elif algorithm == 'hierarchical':
        clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage=linkage_method)
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

        color = plt.cm.tab10(i / optimal_k)

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

    algorithm_title = algorithm
    if algorithm == 'hierarchical':
        algorithm_title = f'{algorithm} ({linkage_method})'

    plt.title(f'Silhouette Analysis ({algorithm_title.capitalize()}, K={optimal_k}) - {dataset_name}')
    plt.xlabel('Silhouette Coefficient Values')
    plt.ylabel('Cluster Label')

    plt.xlim([-0.1, 1])
    plt.ylim([0, y_lower + 10])

    plt.legend(loc='best')
    plt.tight_layout()

    filename_suffix = ''
    if algorithm == 'hierarchical':
        filename_suffix = f'_{linkage_method}'

    plot_filename = os.path.join(
        output_dir,
        f"{dataset_name}_{algorithm}{filename_suffix}_silhouette_analysis_k{optimal_k}.png"
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_cluster_metrics_for_k(features, k, algorithm='kmeans', linkage_method='ward'):
    if algorithm == 'kmeans':
        clustering = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = clustering.fit_predict(features)
        inertia = clustering.inertia_
    elif algorithm == 'hierarchical':
        clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        cluster_labels = clustering.fit_predict(features)
        inertia = None
    else:
        raise ValueError("Algorithm must be 'kmeans' or 'hierarchical'")

    silhouette = silhouette_score(features, cluster_labels) if k > 1 else 0

    metrics = {
        'k': k,
        'algorithm': algorithm,
        'linkage': linkage_method if algorithm == 'hierarchical' else None,
        'inertia': inertia,
        'silhouette': silhouette
    }

    return metrics
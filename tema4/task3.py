import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from task3_preprocess import preprocess_and_select_features
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.utils import resample

visualization_dir = 'task3_visualization'
os.makedirs(visualization_dir, exist_ok=True)

from task2_cluster_evaluation import perform_kmeans_elbow_analysis, perform_silhouette_analysis, \
    perform_hierarchical_elbow_analysis
from task2_visualization import visualize_clusters_pca


def get_contingency_matrix(true_labels, cluster_labels):
    from sklearn.metrics import confusion_matrix

    true_unique = np.unique(true_labels)
    cluster_unique = np.unique(cluster_labels)

    matrix = np.zeros((len(true_unique), len(cluster_unique)), dtype=int)

    for i, true_label in enumerate(true_unique):
        for j, cluster_label in enumerate(cluster_unique):
            matrix[i, j] = np.sum((true_labels == true_label) & (cluster_labels == cluster_label))

    return matrix, true_unique, cluster_unique


def plot_contingency_matrix(contingency_matrix, true_labels, cluster_labels, algorithm, dataset_name, output_dir):
    """Plot the contingency matrix between true labels and cluster assignments"""
    plt.figure(figsize=(10, 8))

    # Create labels for the heatmap
    true_unique = np.unique(true_labels)
    cluster_unique = np.unique(cluster_labels)

    true_labels_str = [f'Class {label}' for label in true_unique]
    cluster_labels_str = [f'Cluster {label}' for label in cluster_unique]

    # Plot the heatmap
    sns.heatmap(contingency_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=cluster_labels_str, yticklabels=true_labels_str)

    plt.title(f'Contingency Matrix - {algorithm} vs True Classes - {dataset_name}')
    plt.xlabel('Cluster Labels')
    plt.ylabel('True Classes')

    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(output_dir, f"{dataset_name}_{algorithm}_contingency_matrix.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Contingency matrix visualization saved to {plot_filename}")


def perform_memory_efficient_hierarchical_clustering(features, k, linkage_method, max_samples=1000):
    n_samples = features.shape[0]

    if n_samples <= max_samples:
        # If the dataset is small enough, use all samples
        hierarchical = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        return hierarchical.fit_predict(features)
    else:
        print(f"Dataset too large for hierarchical clustering ({n_samples} samples).")
        print(f"Using a random subset of {max_samples} samples and KMeans to label the remaining samples.")

        np.random.seed(42)
        subset_indices = np.random.choice(n_samples, max_samples, replace=False)
        X_subset = features[subset_indices]

        hierarchical = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        subset_labels = hierarchical.fit_predict(X_subset)

        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X_subset, subset_labels)

        # Predict labels for the full dataset using the KMeans model
        full_labels = kmeans.predict(features)

        return full_labels


def create_balanced_limited_dataset(X, y, max_samples=5000, random_state=42):
    X = np.array(X)
    y = np.array(y)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    num_classes = len(unique_classes)

    samples_per_class = min(min(class_counts), max_samples // num_classes)

    X_balanced = np.empty((samples_per_class * num_classes, X.shape[1]), dtype=X.dtype)
    y_balanced = np.empty(samples_per_class * num_classes, dtype=y.dtype)

    np.random.seed(random_state)
    start_idx = 0

    for i, cls in enumerate(unique_classes):
        cls_indices = np.where(y == cls)[0]

        if len(cls_indices) > samples_per_class:
            selected_indices = np.random.choice(cls_indices, samples_per_class, replace=False)
        else:
            selected_indices = cls_indices

        end_idx = start_idx + len(selected_indices)
        X_balanced[start_idx:end_idx] = X[selected_indices]
        y_balanced[start_idx:end_idx] = y[selected_indices]

        start_idx = end_idx

    if start_idx < len(X_balanced):
        X_balanced = X_balanced[:start_idx]
        y_balanced = y_balanced[:start_idx]

    return X_balanced, y_balanced


def main():
    MAX_SAMPLES = 5000

    print("Loading airlines delay dataset...")
    try:
        df = pd.read_csv('data/airlines_delay.csv')
        print(f"Dataset shape: {df.shape}")
    except FileNotFoundError:
        print("Error: 'airlines_delay.csv' file not found. Please make sure the file is in the correct directory.")
        return

    try:


        X_selected, y, selected_features, preprocessor, selected_indices = preprocess_and_select_features(
            df, n_features=None, use_routes=True, use_time_features=True, use_interactions=True)

        print(f"Selected {len(selected_features)} features for clustering")
    except ImportError:
        print("Warning: Could not import preprocessing functions. Using a simplified approach.")
        from sklearn.preprocessing import StandardScaler

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'Class' in numeric_cols:
            numeric_cols.remove('Class')

        X = df[numeric_cols].values
        scaler = StandardScaler()
        X_selected = scaler.fit_transform(X)

        y = df['Class']

        print(f"Using {X_selected.shape[1]} numeric features for clustering")

    print("\nOriginal class distribution:")
    class_counts = np.bincount(y.astype(int))
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count} samples ({count / len(y) * 100:.2f}%)")

    X_balanced, y_balanced = create_balanced_limited_dataset(X_selected, y, max_samples=MAX_SAMPLES)

    print(f"\nCreated balanced dataset with {len(y_balanced)} samples (max {MAX_SAMPLES})")

    print("\nBalanced class distribution:")
    balanced_class_counts = np.bincount(y_balanced.astype(int))
    for i, count in enumerate(balanced_class_counts):
        print(f"Class {i}: {count} samples ({count / len(y_balanced) * 100:.2f}%)")

    cluster_range = range(2, 9)

    print("\nPerforming k-means elbow analysis...")
    kmeans_inertias, kmeans_optimal_k = perform_kmeans_elbow_analysis(
        X_balanced,
        cluster_range,
        'airlines_delay_balanced',
        visualization_dir
    )

    linkage_methods = ['ward', 'complete', 'average', 'single']

    print("\nPerforming k-means silhouette analysis...")
    kmeans_silhouette_scores, kmeans_silhouette_optimal_k = perform_silhouette_analysis(
        X_balanced,
        cluster_range,
        'kmeans',
        'airlines_delay_balanced',
        visualization_dir
    )

    optimal_k_values = {
        'KMeans_Elbow': kmeans_optimal_k,
        'KMeans_Silhouette': kmeans_silhouette_optimal_k
    }

    max_hierarchical_size = 1000
    dataset_size = X_balanced.shape[0]
    use_hierarchical = dataset_size <= max_hierarchical_size

    for linkage_method in linkage_methods:
        print(f"\nPerforming hierarchical clustering analysis with {linkage_method} linkage...")

        if use_hierarchical:
            print(f"Running full hierarchical analysis (dataset size: {dataset_size})")
            # Run hierarchical elbow analysis
            hierarchical_distances, hierarchical_optimal_k = perform_hierarchical_elbow_analysis(
                X_balanced,
                cluster_range,
                'airlines_delay_balanced',
                visualization_dir,
                linkage_method
            )

            silhouette_scores, silhouette_optimal_k = perform_silhouette_analysis(
                X_balanced,
                cluster_range,
                'hierarchical',
                'airlines_delay_balanced',
                visualization_dir,
                linkage_method
            )
        else:
            print(f"Dataset too large for full hierarchical analysis ({dataset_size} > {max_hierarchical_size})")
            print("Using K values from K-means analysis as estimates")
            hierarchical_optimal_k = kmeans_optimal_k
            silhouette_optimal_k = kmeans_silhouette_optimal_k

        optimal_k_values[f'Hierarchical_{linkage_method}_Elbow'] = hierarchical_optimal_k
        optimal_k_values[f'Hierarchical_{linkage_method}_Silhouette'] = silhouette_optimal_k
    print("\nSummary of optimal K values:")
    for method, k in optimal_k_values.items():
        print(f"{method}: K = {k}")

    print("\nRunning clustering with optimal K values and calculating ARI...")

    true_labels = y_balanced

    ari_results = {}

    print(f"\nRunning K-means with K = {kmeans_optimal_k} (Elbow method)...")
    kmeans_elbow = KMeans(n_clusters=kmeans_optimal_k, random_state=42, n_init=10)
    kmeans_elbow_labels = kmeans_elbow.fit_predict(X_balanced)
    kmeans_elbow_ari = adjusted_rand_score(true_labels, kmeans_elbow_labels)
    ari_results['KMeans_Elbow'] = kmeans_elbow_ari

    contingency_matrix, _, _ = get_contingency_matrix(true_labels, kmeans_elbow_labels)
    plot_contingency_matrix(contingency_matrix, true_labels, kmeans_elbow_labels,
                            'KMeans_Elbow', 'airlines_delay_balanced', visualization_dir)

    visualize_clusters_pca(X_balanced, kmeans_elbow_labels, kmeans_optimal_k,
                           'KMeans_Elbow', 'airlines_delay_balanced', visualization_dir)

    print(f"\nRunning K-means with K = {kmeans_silhouette_optimal_k} (Silhouette method)...")
    kmeans_silhouette = KMeans(n_clusters=kmeans_silhouette_optimal_k, random_state=42, n_init=10)
    kmeans_silhouette_labels = kmeans_silhouette.fit_predict(X_balanced)
    kmeans_silhouette_ari = adjusted_rand_score(true_labels, kmeans_silhouette_labels)
    ari_results['KMeans_Silhouette'] = kmeans_silhouette_ari

    contingency_matrix, _, _ = get_contingency_matrix(true_labels, kmeans_silhouette_labels)
    plot_contingency_matrix(contingency_matrix, true_labels, kmeans_silhouette_labels,
                            'KMeans_Silhouette', 'airlines_delay_balanced', visualization_dir)

    visualize_clusters_pca(X_balanced, kmeans_silhouette_labels, kmeans_silhouette_optimal_k,
                           'KMeans_Silhouette', 'airlines_delay_balanced', visualization_dir)

    for linkage_method in linkage_methods:
        elbow_k = optimal_k_values[f'Hierarchical_{linkage_method}_Elbow']
        print(f"\nRunning Hierarchical clustering ({linkage_method}) with K = {elbow_k} (Elbow method)...")

        hierarchical_elbow_labels = perform_memory_efficient_hierarchical_clustering(
            X_balanced, elbow_k, linkage_method, max_samples=max_hierarchical_size
        )

        hierarchical_elbow_ari = adjusted_rand_score(true_labels, hierarchical_elbow_labels)
        ari_results[f'Hierarchical_{linkage_method}_Elbow'] = hierarchical_elbow_ari

        contingency_matrix, _, _ = get_contingency_matrix(true_labels, hierarchical_elbow_labels)
        plot_contingency_matrix(contingency_matrix, true_labels, hierarchical_elbow_labels,
                                f'Hierarchical_{linkage_method}_Elbow', 'airlines_delay_balanced', visualization_dir)

        visualize_clusters_pca(X_balanced, hierarchical_elbow_labels, elbow_k,
                               f'Hierarchical_{linkage_method}_Elbow', 'airlines_delay_balanced', visualization_dir)

        silhouette_k = optimal_k_values[f'Hierarchical_{linkage_method}_Silhouette']
        print(f"\nRunning Hierarchical clustering ({linkage_method}) with K = {silhouette_k} (Silhouette method)...")

        hierarchical_silhouette_labels = perform_memory_efficient_hierarchical_clustering(
            X_balanced, silhouette_k, linkage_method, max_samples=max_hierarchical_size
        )

        hierarchical_silhouette_ari = adjusted_rand_score(true_labels, hierarchical_silhouette_labels)
        ari_results[f'Hierarchical_{linkage_method}_Silhouette'] = hierarchical_silhouette_ari

        contingency_matrix, _, _ = get_contingency_matrix(true_labels, hierarchical_silhouette_labels)
        plot_contingency_matrix(contingency_matrix, true_labels, hierarchical_silhouette_labels,
                                f'Hierarchical_{linkage_method}_Silhouette', 'airlines_delay_balanced',
                                visualization_dir)

        visualize_clusters_pca(X_balanced, hierarchical_silhouette_labels, silhouette_k,
                               f'Hierarchical_{linkage_method}_Silhouette', 'airlines_delay_balanced',
                               visualization_dir)

    print("\nARI Results Summary:")
    for method, ari in ari_results.items():
        print(f"{method}: ARI = {ari:.4f}")

    best_method = max(ari_results, key=ari_results.get)
    best_ari = ari_results[best_method]
    best_k = optimal_k_values[best_method]

    print(f"\nBest clustering method: {best_method}")
    print(f"Best K: {best_k}")
    print(f"Best ARI: {best_ari:.4f}")

    plt.figure(figsize=(12, 8))

    sorted_methods = sorted(ari_results.items(), key=lambda x: x[1], reverse=True)
    methods = [method for method, _ in sorted_methods]
    ari_values = [ari for _, ari in sorted_methods]

    bars = plt.bar(methods, ari_values, color='skyblue')

    best_idx = methods.index(best_method)
    bars[best_idx].set_color('red')

    plt.xlabel('Clustering Method')
    plt.ylabel('Adjusted Rand Index (ARI)')
    plt.title('Comparison of Clustering Methods by ARI - Airlines Delay Dataset (Balanced)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(ari_values) * 1.1)

    for i, v in enumerate(ari_values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'airlines_delay_balanced_ari_comparison.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))

    methods = list(optimal_k_values.keys())
    k_values = [optimal_k_values[method] for method in methods]

    bars = plt.bar(methods, k_values, color='lightgreen')

    best_idx = methods.index(best_method)
    bars[best_idx].set_color('red')

    plt.xlabel('Clustering Method')
    plt.ylabel('Optimal Number of Clusters')
    plt.title('Optimal K Values by Method - Airlines Delay Dataset (Balanced)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(k_values) * 1.1)

    for i, v in enumerate(k_values):
        plt.text(i, v + 0.1, str(v), ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'airlines_delay_balanced_optimal_k_comparison.png'), dpi=300)
    plt.close()

    print(f"\nSummary visualizations saved to {visualization_dir}")

    results_df = pd.DataFrame({
        'Method': list(ari_results.keys()),
        'Optimal_K': [optimal_k_values[method] for method in ari_results.keys()],
        'ARI': list(ari_results.values())
    })
    results_df.sort_values('ARI', ascending=False, inplace=True)
    results_df.to_csv(os.path.join(visualization_dir, 'airlines_delay_balanced_clustering_results.csv'), index=False)
    print(f"Results saved to {os.path.join(visualization_dir, 'airlines_delay_balanced_clustering_results.csv')}")


if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from task1_data_prep import read_data_and_num_clusters
from task1_clustering_algorithms import normalize_data
from task2_cluster_evaluation import *
from task2_visualization import visualize_clusters_pca, plot_silhouette_visualization

visualization_dir = 'task2_visualization'
os.makedirs(visualization_dir, exist_ok=True)

data_files = read_data_and_num_clusters('data')

target_datasets = [name for name in data_files.keys() if 'iris' in name.lower() or '2d-10c' in name.lower()]

if not target_datasets:
    print("Error: Could not find Iris or 2d-10c datasets in the data directory.")
    exit(1)

linkage_methods = ['ward', 'complete', 'average', 'single']

results = []

for file_name in target_datasets:
    print(f"\n{'=' * 50}")
    print(f"Processing dataset: {file_name}")
    print(f"{'=' * 50}")

    features = data_files[file_name]['features']
    true_labels = data_files[file_name]['labels']
    true_num_clusters = data_files[file_name]['num_clusters']

    normalized_features = normalize_data(features)

    print(f"True number of clusters: {true_num_clusters}")

    if '2d-10c' in file_name:
        cluster_range = range(2, 15)
        print(f"Using extended cluster range (2-11) for {file_name}")
    else:
        cluster_range = range(2, 11)  #
        print(f"Using default cluster range (2-10) for {file_name}")

    print("\nPerforming k-means elbow analysis...")
    kmeans_inertias, kmeans_optimal_k = perform_kmeans_elbow_analysis(
        normalized_features,
        cluster_range,
        file_name,
        visualization_dir
    )

    print("\nPerforming k-means silhouette analysis...")
    kmeans_silhouette_scores, kmeans_silhouette_optimal_k = perform_silhouette_analysis(
        normalized_features,
        cluster_range,
        'kmeans',
        file_name,
        visualization_dir
    )

    plot_silhouette_visualization(
        normalized_features,
        kmeans_silhouette_optimal_k,
        'kmeans',
        file_name,
        visualization_dir
    )

    result = {
        'Dataset': file_name,
        'True_Clusters': true_num_clusters,
        'KMeans_Elbow_Optimal_K': kmeans_optimal_k,
        'KMeans_Silhouette_Optimal_K': kmeans_silhouette_optimal_k
    }

    for linkage_method in linkage_methods:
        print(f"\nPerforming hierarchical clustering analysis with {linkage_method} linkage...")

        print(f"Performing hierarchical elbow analysis ({linkage_method})...")
        hierarchical_distances, hierarchical_optimal_k = perform_hierarchical_elbow_analysis(
            normalized_features,
            cluster_range,
            file_name,
            visualization_dir,
            linkage_method
        )

        print(f"Performing hierarchical silhouette analysis ({linkage_method})...")
        hierarchical_silhouette_scores, hierarchical_silhouette_optimal_k = perform_silhouette_analysis(
            normalized_features,
            cluster_range,
            'hierarchical',
            file_name,
            visualization_dir,
            linkage_method
        )

        plot_silhouette_visualization(
            normalized_features,
            hierarchical_silhouette_optimal_k,
            'hierarchical',
            file_name,
            visualization_dir,
            linkage_method
        )

        result[f'Hierarchical_{linkage_method}_Elbow_Optimal_K'] = hierarchical_optimal_k
        result[f'Hierarchical_{linkage_method}_Silhouette_Optimal_K'] = hierarchical_silhouette_optimal_k

    results.append(result)

    print("\nResults summary for dataset:", file_name)
    print(f"True number of clusters: {true_num_clusters}")
    print(f"K-means elbow method optimal K: {result['KMeans_Elbow_Optimal_K']}")
    print(f"K-means silhouette method optimal K: {result['KMeans_Silhouette_Optimal_K']}")

    for linkage_method in linkage_methods:
        print(
            f"Hierarchical ({linkage_method}) elbow method optimal K: {result[f'Hierarchical_{linkage_method}_Elbow_Optimal_K']}")
        print(
            f"Hierarchical ({linkage_method}) silhouette method optimal K: {result[f'Hierarchical_{linkage_method}_Silhouette_Optimal_K']}")

results_df = pd.DataFrame(results)
output_filename = 'task2_results.csv'
results_df.to_csv(output_filename, index=False)
print(f"\nResults saved to {output_filename}")

print("\nOverall summary:")
print(results_df.to_string())

plt.figure(figsize=(15, 10))

n_datasets = len(results)
n_methods = 1 + 1 + len(linkage_methods) * 2
bar_width = 0.8 / n_methods
index = np.arange(n_datasets)

method_colors = {
    'True': 'black',
    'KMeans_Elbow': 'blue',
    'KMeans_Silhouette': 'royalblue',
    'Hierarchical_ward_Elbow': 'green',
    'Hierarchical_ward_Silhouette': 'lightgreen',
    'Hierarchical_complete_Elbow': 'red',
    'Hierarchical_complete_Silhouette': 'salmon',
    'Hierarchical_average_Elbow': 'purple',
    'Hierarchical_average_Silhouette': 'plum',
    'Hierarchical_single_Elbow': 'orange',
    'Hierarchical_single_Silhouette': 'gold'
}

offset = -n_methods * bar_width / 2
plt.bar(index + offset, [r['True_Clusters'] for r in results], bar_width,
        label='True Clusters', color=method_colors['True'])
offset += bar_width

plt.bar(index + offset, [r['KMeans_Elbow_Optimal_K'] for r in results], bar_width,
        label='KMeans Elbow', color=method_colors['KMeans_Elbow'])
offset += bar_width

plt.bar(index + offset, [r['KMeans_Silhouette_Optimal_K'] for r in results], bar_width,
        label='KMeans Silhouette', color=method_colors['KMeans_Silhouette'])
offset += bar_width

for linkage_method in linkage_methods:
    plt.bar(index + offset,
            [r[f'Hierarchical_{linkage_method}_Elbow_Optimal_K'] for r in results],
            bar_width,
            label=f'Hierarchical {linkage_method} Elbow',
            color=method_colors[f'Hierarchical_{linkage_method}_Elbow'])
    offset += bar_width

    plt.bar(index + offset,
            [r[f'Hierarchical_{linkage_method}_Silhouette_Optimal_K'] for r in results],
            bar_width,
            label=f'Hierarchical {linkage_method} Silhouette',
            color=method_colors[f'Hierarchical_{linkage_method}_Silhouette'])
    offset += bar_width

plt.xlabel('Dataset')
plt.ylabel('Number of Clusters')
plt.title('Comparison of Estimated vs True Number of Clusters')
plt.xticks(index, [r['Dataset'] for r in results])
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig(os.path.join(visualization_dir, 'cluster_number_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nComparative analysis chart saved to {os.path.join(visualization_dir, 'cluster_number_comparison.png')}")

print("\nGenerating PCA visualizations for optimal clustering results...")
from sklearn.cluster import KMeans, AgglomerativeClustering

for r in results:
    dataset = r['Dataset']
    features = data_files[dataset]['features']
    normalized_features = normalize_data(features)

    print(f"\nGenerating visualizations for dataset: {dataset}")

    for method in ['Elbow', 'Silhouette']:
        k = r[f'KMeans_{method}_Optimal_K']
        print(f"  KMeans clustering with {method} method (K={k})...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized_features)

        visualize_clusters_pca(
            normalized_features,
            labels,
            k,
            'KMeans',
            dataset,
            visualization_dir,
            title_suffix=f' ({method} Method)'
        )

    for linkage_method in linkage_methods:
        for method in ['Elbow', 'Silhouette']:
            k = r[f'Hierarchical_{linkage_method}_{method}_Optimal_K']
            print(f"  Hierarchical clustering with {linkage_method} linkage, {method} method (K={k})...")

            hierarchical = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
            labels = hierarchical.fit_predict(normalized_features)

            visualize_clusters_pca(
                normalized_features,
                labels,
                k,
                f'Hierarchical_{linkage_method}',
                dataset,
                visualization_dir,
                title_suffix=f' ({method} Method)'
            )

    true_labels = data_files[dataset]['labels']
    true_k = len(np.unique(true_labels))

    print(f"  True labels visualization (K={true_k})...")
    visualize_clusters_pca(
        normalized_features,
        true_labels,
        true_k,
        'True',
        dataset,
        visualization_dir,
        title_suffix=' (Ground Truth)'
    )

print("\nAll visualizations generated successfully!")

print("\nComprehensive Results for Each Dataset:")
for r in results:
    dataset = r['Dataset']
    true_k = r['True_Clusters']

    print(f"\n{'-' * 30}")
    print(f"Dataset: {dataset}")
    print(f"True number of clusters: {true_k}")
    print(f"{'-' * 30}")

    methods = {}

    kmeans_elbow_k = r['KMeans_Elbow_Optimal_K']
    kmeans_elbow_error = abs(kmeans_elbow_k - true_k)
    methods['KMeans_Elbow'] = kmeans_elbow_error

    kmeans_silhouette_k = r['KMeans_Silhouette_Optimal_K']
    kmeans_silhouette_error = abs(kmeans_silhouette_k - true_k)
    methods['KMeans_Silhouette'] = kmeans_silhouette_error

    print(f"K-means Elbow Method: K = {kmeans_elbow_k}, Error = {kmeans_elbow_error}")
    print(f"K-means Silhouette Method: K = {kmeans_silhouette_k}, Error = {kmeans_silhouette_error}")

    for linkage_method in linkage_methods:
        h_elbow_k = r[f'Hierarchical_{linkage_method}_Elbow_Optimal_K']
        h_elbow_error = abs(h_elbow_k - true_k)
        methods[f'Hierarchical_{linkage_method}_Elbow'] = h_elbow_error

        h_silhouette_k = r[f'Hierarchical_{linkage_method}_Silhouette_Optimal_K']
        h_silhouette_error = abs(h_silhouette_k - true_k)
        methods[f'Hierarchical_{linkage_method}_Silhouette'] = h_silhouette_error

        print(f"Hierarchical ({linkage_method}) Elbow Method: K = {h_elbow_k}, Error = {h_elbow_error}")
        print(f"Hierarchical ({linkage_method}) Silhouette Method: K = {h_silhouette_k}, Error = {h_silhouette_error}")

    # Find the best method
    best_method = min(methods, key=methods.get)
    best_value = r[best_method + "_Optimal_K"]  # Fixed key lookup

    print(f"\n>>> BEST METHOD: {best_method}")
    print(f">>> Optimal K = {best_value}, Error = {methods[best_method]}")

# Method ranking by average error
print("\nMethod ranking by average error:")
method_avg_errors = {}

for method_name in ['KMeans_Elbow', 'KMeans_Silhouette']:
    total_error = sum(abs(r[f'{method_name}_Optimal_K'] - r['True_Clusters']) for r in results)
    method_avg_errors[method_name] = total_error / len(results)

for linkage in linkage_methods:
    for method_type in ['Elbow', 'Silhouette']:
        method_name = f'Hierarchical_{linkage}_{method_type}'
        key = f'Hierarchical_{linkage}_{method_type}_Optimal_K'
        total_error = sum(abs(r[key] - r['True_Clusters']) for r in results)
        method_avg_errors[method_name] = total_error / len(results)

sorted_methods = sorted(method_avg_errors.items(), key=lambda x: x[1])

for i, (method, avg_error) in enumerate(sorted_methods, 1):
    print(f"{i}. {method}: Average error = {avg_error:.2f}")

error_rows = []
for r in results:
    row = {'Dataset': r['Dataset'], 'True_Clusters': r['True_Clusters']}

    row['KMeans_Elbow_Error'] = abs(r['KMeans_Elbow_Optimal_K'] - r['True_Clusters'])
    row['KMeans_Silhouette_Error'] = abs(r['KMeans_Silhouette_Optimal_K'] - r['True_Clusters'])

    for linkage in linkage_methods:
        row[f'Hierarchical_{linkage}_Elbow_Error'] = abs(
            r[f'Hierarchical_{linkage}_Elbow_Optimal_K'] - r['True_Clusters'])
        row[f'Hierarchical_{linkage}_Silhouette_Error'] = abs(
            r[f'Hierarchical_{linkage}_Silhouette_Optimal_K'] - r['True_Clusters'])

    error_rows.append(row)

error_summary = pd.DataFrame(error_rows)
error_summary_filename = 'task2_error_analysis.csv'
error_summary.to_csv(error_summary_filename, index=False)
print(f"\nError analysis saved to {error_summary_filename}")
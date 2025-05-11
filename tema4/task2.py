import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from task1_data_prep import read_data_and_num_clusters
from task1_clustering_algorithms import normalize_data
from task2_cluster_evaluation import *
from task2_visualization import *


visualization_dir = 'task2_visualization'
os.makedirs(visualization_dir, exist_ok=True)

data_files = read_data_and_num_clusters('data')

# Filter only Iris and 2d-10c datasets
target_datasets = [name for name in data_files.keys() if 'iris' in name.lower() or '2d-10c' in name.lower()]

if not target_datasets:
    print("Error: Could not find Iris or 2d-10c datasets in the data directory.")
    exit(1)

# Results container
results = []

# Process each target dataset
for file_name in target_datasets:
    print(f"\n{'=' * 50}")
    print(f"Processing dataset: {file_name}")
    print(f"{'=' * 50}")

    # Get data
    features = data_files[file_name]['features']
    true_labels = data_files[file_name]['labels']
    true_num_clusters = data_files[file_name]['num_clusters']

    # Normalize data
    normalized_features = normalize_data(features)

    print(f"True number of clusters: {true_num_clusters}")

    # Define cluster range to test - from 2 to 10
    cluster_range = range(2, 11)

    # Perform k-means elbow analysis
    print("\nPerforming k-means elbow analysis...")
    kmeans_inertias, kmeans_optimal_k = perform_kmeans_elbow_analysis(
        normalized_features,
        cluster_range,
        file_name,
        visualization_dir
    )

    # Perform hierarchical elbow analysis
    print("\nPerforming hierarchical elbow analysis...")
    hierarchical_distances, hierarchical_optimal_k = perform_hierarchical_elbow_analysis(
        normalized_features,
        cluster_range,
        file_name,
        visualization_dir
    )

    # Perform silhouette analysis for both methods
    print("\nPerforming silhouette analysis...")
    kmeans_silhouette_scores, kmeans_silhouette_optimal_k = perform_silhouette_analysis(
        normalized_features,
        cluster_range,
        'kmeans',
        file_name,
        visualization_dir
    )

    hierarchical_silhouette_scores, hierarchical_silhouette_optimal_k = perform_silhouette_analysis(
        normalized_features,
        cluster_range,
        'hierarchical',
        file_name,
        visualization_dir
    )

    # Collect results
    result = {
        'Dataset': file_name,
        'True_Clusters': true_num_clusters,
        'KMeans_Elbow_Optimal_K': kmeans_optimal_k,
        'Hierarchical_Elbow_Optimal_K': hierarchical_optimal_k,
        'KMeans_Silhouette_Optimal_K': kmeans_silhouette_optimal_k,
        'Hierarchical_Silhouette_Optimal_K': hierarchical_silhouette_optimal_k
    }

    results.append(result)

    print("\nResults summary:")
    print(f"True number of clusters: {true_num_clusters}")
    print(f"K-means elbow method optimal K: {kmeans_optimal_k}")
    print(f"Hierarchical elbow method optimal K: {hierarchical_optimal_k}")
    print(f"K-means silhouette method optimal K: {kmeans_silhouette_optimal_k}")
    print(f"Hierarchical silhouette method optimal K: {hierarchical_silhouette_optimal_k}")

    # Visualize clustering results for optimal K values
    method_results = {
        'KMeans_Elbow': kmeans_optimal_k,
        'Hierarchical_Elbow': hierarchical_optimal_k,
        'KMeans_Silhouette': kmeans_silhouette_optimal_k,
        'Hierarchical_Silhouette': hierarchical_silhouette_optimal_k
    }

    print("\nGenerating cluster visualizations...")
    visualize_optimal_clustering_results(
        normalized_features,
        method_results,
        file_name,
        visualization_dir
    )

    # Create detailed silhouette visualizations for the best methods
    print("\nGenerating silhouette visualizations...")
    plot_silhouette_visualization(
        normalized_features,
        kmeans_silhouette_optimal_k,
        'kmeans',
        file_name,
        visualization_dir
    )

    plot_silhouette_visualization(
        normalized_features,
        hierarchical_silhouette_optimal_k,
        'hierarchical',
        file_name,
        visualization_dir
    )

# Save results to CSV
results_df = pd.DataFrame(results)
output_filename = 'task2_results.csv'
results_df.to_csv(output_filename, index=False)
print(f"\nResults saved to {output_filename}")

# Print overall summary
print("\nOverall summary:")
print(results_df)

# Create a comparative bar chart of estimated vs true cluster numbers
plt.figure(figsize=(12, 8))

# Set up bar positions
n_datasets = len(results)
bar_width = 0.15
index = np.arange(n_datasets)

# Plot bars for each method
plt.bar(index - 2 * bar_width, [r['True_Clusters'] for r in results], bar_width,
        label='True Clusters', color='black')
plt.bar(index - bar_width, [r['KMeans_Elbow_Optimal_K'] for r in results], bar_width,
        label='KMeans Elbow', color='blue')
plt.bar(index, [r['Hierarchical_Elbow_Optimal_K'] for r in results], bar_width,
        label='Hierarchical Elbow', color='green')
plt.bar(index + bar_width, [r['KMeans_Silhouette_Optimal_K'] for r in results], bar_width,
        label='KMeans Silhouette', color='red')
plt.bar(index + 2 * bar_width, [r['Hierarchical_Silhouette_Optimal_K'] for r in results], bar_width,
        label='Hierarchical Silhouette', color='orange')

# Labels and formatting
plt.xlabel('Dataset')
plt.ylabel('Number of Clusters')
plt.title('Comparison of Estimated vs True Number of Clusters')
plt.xticks(index, [r['Dataset'] for r in results])
plt.legend()
plt.tight_layout()

# Save comparative chart
plt.savefig(os.path.join(visualization_dir, 'cluster_number_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nComparative analysis chart saved to {os.path.join(visualization_dir, 'cluster_number_comparison.png')}")
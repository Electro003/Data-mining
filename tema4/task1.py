import os
import pandas as pd
from task1_clustering_algorithms import *
from tema4.task1_data_prep import read_data_and_num_clusters
from tema4.task1_iris_som import apply_multiple_som_configs


def save_ari_scores_to_csv(ari_results, output_filename='task1_ARI_results.csv'):
    ari_df = pd.DataFrame(ari_results)
    ari_df.to_csv(output_filename, index=False)
    print(f"ARI results saved to {output_filename}")


data_files = read_data_and_num_clusters('data')

visualization_dir = 'task1_visualization'
os.makedirs(visualization_dir, exist_ok=True)

# List to store ARI results
ari_results = []

for file_name, data in data_files.items():
    features = data['features']
    labels = data['labels']
    num_clusters = data['num_clusters']

    print(f"Processing {file_name} with {num_clusters} clusters...")

    normalized_features = normalize_data(features)

    file_ari_scores = {'File': file_name}

    # Apply kMeans
    kmeans_labels, kmeans_model = apply_kmeans(normalized_features, num_clusters)
    kmeans_ari = compute_ari(labels, kmeans_labels)
    file_ari_scores['kMeans'] = kmeans_ari
    kmeans_filename = os.path.join(visualization_dir, f"{file_name}_kmeans.png")
    apply_pca_and_plot(normalized_features, labels, kmeans_labels, 'KMeans Clustering', f'{file_name}_kmeans')
    print(f"kMeans ARI for {file_name}: {kmeans_ari}")

    # Apply GMM
    gmm_labels, gmm_model = apply_gmm(normalized_features, num_clusters)
    gmm_ari = compute_ari(labels, gmm_labels)
    file_ari_scores['GMM'] = gmm_ari
    gmm_filename = os.path.join(visualization_dir, f"{file_name}_gmm.png")
    apply_pca_and_plot(normalized_features, labels, gmm_labels, 'Gaussian Mixture Model Clustering', f'{file_name}_gmm')
    print(f"GMM ARI for {file_name}: {gmm_ari}")

    # Apply Hierarchical clustering with different linkage methods
    for linkage in ['ward', 'single', 'complete', 'average']:
        hierarchical_labels = apply_hierarchical(normalized_features, num_clusters, linkage)
        hierarchical_ari = compute_ari(labels, hierarchical_labels)
        file_ari_scores[f'Hierarchical_{linkage}'] = hierarchical_ari
        hierarchical_filename = os.path.join(visualization_dir, f"{file_name}_hierarchical_{linkage}.png")
        apply_pca_and_plot(normalized_features, labels, hierarchical_labels, f'Hierarchical Clustering ({linkage} Link)', f'{file_name}_hierarchical_{linkage}')
        print(f"Hierarchical ARI for {file_name} ({linkage}): {hierarchical_ari}")

    # Apply DBSCAN
    dbscan_labels = apply_dbscan(normalized_features)
    dbscan_ari = compute_ari(labels, dbscan_labels)
    file_ari_scores['DBSCAN'] = dbscan_ari
    dbscan_filename = os.path.join(visualization_dir, f"{file_name}_dbscan.png")
    apply_pca_and_plot(normalized_features, labels, dbscan_labels, 'DBSCAN Clustering', f'{file_name}_dbscan')
    print(f"DBSCAN ARI for {file_name}: {dbscan_ari}")

    if 'iris' in file_name.lower():
        som_labels, som_model, som_ari= apply_multiple_som_configs(normalized_features, labels, file_name)
        file_ari_scores['SOM'] = som_ari
        apply_pca_and_plot(normalized_features, labels, som_labels, 'SOM Clustering', f'{file_name}_som_pca')
    ari_results.append(file_ari_scores)

save_ari_scores_to_csv(ari_results)

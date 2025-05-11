import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from sklearn.metrics import adjusted_rand_score


def visualize_som_map(features, labels, som_model, map_size, title):
    plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

    # Plot 1: U-Matrix (Unified Distance Matrix)
    ax1 = plt.subplot(gs[0, 0])
    umatrix = som_model.distance_map()
    ax1.imshow(umatrix, cmap='bone_r')
    ax1.set_title('U-Matrix (Distance Map)', fontsize=14)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = plt.subplot(gs[0, 1])
    hit_map = np.zeros((map_size[0], map_size[1], 3))

    win_map = {}
    for i, x in enumerate(features):
        win = som_model.winner(x)
        if win not in win_map:
            win_map[win] = []
        win_map[win].append(i)

    colors = ['#FF5733', '#33FF57', '#3357FF']

    color_tuples = []
    for hex_color in colors:
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
        color_tuples.append(rgb)

    class_colors = {i: color_tuples[i] for i in range(3)}

    for position, samples in win_map.items():
        class_counts = np.zeros(3)
        for sample_idx in samples:
            class_counts[labels[sample_idx]] += 1

        if len(samples) > 0:
            dominant_class = np.argmax(class_counts)
            hit_map[position[0], position[1]] = class_colors[dominant_class]

    ax2.imshow(hit_map)
    ax2.set_title('Sample Hits by Class', fontsize=14)
    ax2.set_xticks([])
    ax2.set_yticks([])

    class_names = ['Setosa', 'Versicolor', 'Virginica']
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colors[i], label=class_names[i])
                       for i in range(3)]
    ax2.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, -0.05), ncol=3)

    # Plot 3: Hits as a heatmap (frequency)
    ax3 = plt.subplot(gs[0, 2])
    hit_count = np.zeros((map_size[0], map_size[1]))
    for position, samples in win_map.items():
        hit_count[position[0], position[1]] = len(samples)

    im = ax3.imshow(hit_count, cmap='viridis')
    ax3.set_title('Hit Frequency', fontsize=14)
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    for i, feature_name in enumerate(feature_names):
        ax = plt.subplot(gs[1, i % 3])
        component_plane = som_model.get_weights()[:, :, i]
        im = ax.imshow(component_plane, cmap='coolwarm')
        ax.set_title(f'Component Plane: {feature_name}', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Set main title
    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return plt.gcf()


def visualize_som_clusters(features, true_labels, som_labels, title, map_size, som_model):
    ari = adjusted_rand_score(true_labels, som_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    win_coords = np.array([som_model.winner(x) for x in features])


    colors = ['#FF5733', '#33FF57', '#3357FF']
    cmap_true = ListedColormap(colors)

    scatter1 = ax1.scatter(
        win_coords[:, 1], win_coords[:, 0],
        c=true_labels,
        cmap=cmap_true,
        s=100,
        alpha=0.8,
        edgecolors='w'
    )

    ax1.set_xticks(np.arange(0, map_size[1], 1))
    ax1.set_yticks(np.arange(0, map_size[0], 1))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.grid(True)
    ax1.set_title('True Labels', fontsize=14)


    cmap_pred = ListedColormap([colors[i] for i in range(3)])

    scatter2 = ax2.scatter(
        win_coords[:, 1], win_coords[:, 0],
        c=som_labels,
        cmap=cmap_pred,
        s=100,
        alpha=0.8,
        edgecolors='w'
    )

    ax2.set_xticks(np.arange(0, map_size[1], 1))
    ax2.set_yticks(np.arange(0, map_size[0], 1))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.grid(True)
    ax2.set_title('SOM+kMeans Clusters', fontsize=14)

    class_names = ['Setosa', 'Versicolor', 'Virginica']
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=colors[i], markersize=10, label=class_names[i])
                       for i in range(3)]

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
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

    return fig


def apply_multiple_som_configs(features, labels, file_name):
    from sklearn.metrics import adjusted_rand_score
    from task1_clustering_algorithms import apply_som

    results = {}

    # Configuration 1: Small map (5x5)
    map_size1 = (5, 5)
    print(f"Running SOM with map size {map_size1}...")
    som_labels1, som_model1 = apply_som(features, map_size=map_size1)
    som_ari1 = adjusted_rand_score(labels, som_labels1)
    print(f"SOM+kMeans with {map_size1} map - ARI: {som_ari1:.4f}")

    # Visualize SOM components and clusters for config 1
    fig1 = visualize_som_map(features, labels, som_model1, map_size1,
                             f"SOM Analysis - Iris Dataset (Map Size: {map_size1[0]}x{map_size1[1]})")
    fig1.savefig(f"task1_visualization/{file_name}_som_map_5x5.png", dpi=300, bbox_inches='tight')

    fig2 = visualize_som_clusters(features, labels, som_labels1,
                                  f"SOM+kMeans Clustering (Map: {map_size1[0]}x{map_size1[1]})",
                                  map_size1, som_model1)
    fig2.savefig(f"task1_visualization/{file_name}_som_clusters_5x5.png", dpi=300, bbox_inches='tight')

    # Configuration 2: Larger map (10x10)
    map_size2 = (10, 10)
    print(f"Running SOM with map size {map_size2}...")
    som_labels2, som_model2 = apply_som(features, map_size=map_size2)
    som_ari2 = adjusted_rand_score(labels, som_labels2)
    print(f"SOM+kMeans with {map_size2} map - ARI: {som_ari2:.4f}")

    # Visualize SOM components and clusters for config 2
    fig3 = visualize_som_map(features, labels, som_model2, map_size2,
                             f"SOM Analysis - Iris Dataset (Map Size: {map_size2[0]}x{map_size2[1]})")
    fig3.savefig(f"task1_visualization/{file_name}_som_map_10x10.png", dpi=300, bbox_inches='tight')

    fig4 = visualize_som_clusters(features, labels, som_labels2,
                                  f"SOM+kMeans Clustering (Map: {map_size2[0]}x{map_size2[1]})",
                                  map_size2, som_model2)
    fig4.savefig(f"task1_visualization/{file_name}_som_clusters_10x10.png", dpi=300, bbox_inches='tight')

    results = {
        'config1': {'map_size': map_size1, 'labels': som_labels1, 'model': som_model1, 'ari': som_ari1},
        'config2': {'map_size': map_size2, 'labels': som_labels2, 'model': som_model2, 'ari': som_ari2}
    }

    best_config = max(results.values(), key=lambda x: x['ari'])

    print("\nSOM+kMeans Results Summary:")
    print(f"Map Size {map_size1}: ARI = {som_ari1:.4f}")
    print(f"Map Size {map_size2}: ARI = {som_ari2:.4f}")
    print(f"Best configuration: Map {best_config['map_size']} with ARI = {best_config['ari']:.4f}")

    return best_config['labels'], best_config['model'], best_config['ari']
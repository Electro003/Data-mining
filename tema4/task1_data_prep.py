import pandas as pd
import os


def read_data_and_num_clusters(data_dir='data'):
    files = [f for f in os.listdir(data_dir)]
    data_files = {}

    for file in files:
        file_path = os.path.join(data_dir, file)

        if file.endswith(('.dat', '.data')):
            df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        else:
            df = pd.read_csv(file_path, header=None)

        feature_columns = [f"feature_{i}" for i in range(df.shape[1] - 1)]
        df.columns = feature_columns + ['label']

        features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values

        num_clusters = len(set(labels))

        data_files[file] = {
            'features': features,
            'labels': labels,
            'num_clusters': num_clusters,
            'df': df
        }

    return data_files

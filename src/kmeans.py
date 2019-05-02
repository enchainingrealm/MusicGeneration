import os
import pickle
from datetime import datetime
from sys import maxsize

import numpy as np
from sklearn.cluster import KMeans

from src.constants import N_CLUSTERS, MODELS_DIRPATH


def main():
    start_time = datetime.now()

    chunks, X = load_features()

    print("Started training K-Means.")

    kmeans = KMeans(
        n_clusters=N_CLUSTERS, n_init=4, max_iter=300, verbose=maxsize,
        random_state=1731, n_jobs=-1
    )
    kmeans.fit(X)

    print("Finished training K-Means.    "
          f"Elapsed time {datetime.now() - start_time}")

    save_cluster_assignments(chunks, kmeans.labels_)
    save_cluster_means(kmeans.cluster_centers_)


def load_features():
    with open(os.path.join(MODELS_DIRPATH, "rescale.pkl"), "rb") as file:
        features_dict = pickle.load(file)

    chunks = []
    Xs = []
    for chunk, features in features_dict.items():
        chunks.append(chunk)
        Xs.append(features.to_row_array())
    X = np.concatenate(Xs, axis=0)

    print(X.shape)
    return chunks, X


def save_cluster_assignments(chunks, labels):
    chunk_to_cluster = {
        chunk: cluster
        for chunk, cluster in zip(chunks, labels)
    }

    cluster_to_chunks = {}
    for chunk, cluster in zip(chunks, labels):
        if cluster not in cluster_to_chunks:
            cluster_to_chunks[cluster] = set()

        cluster_to_chunks[cluster].add(chunk)

    with open(os.path.join(
            MODELS_DIRPATH, "kmeans_chunk_to_cluster.pkl"
    ), "wb") as file:
        pickle.dump(chunk_to_cluster, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(
            MODELS_DIRPATH, "kmeans_cluster_to_chunks.pkl"
    ), "wb") as file:
        pickle.dump(cluster_to_chunks, file, protocol=pickle.HIGHEST_PROTOCOL)


def save_cluster_means(cluster_centers):
    result = {
        cluster: cluster_centers[cluster, :]
        for cluster in range(N_CLUSTERS)
    }

    with open(os.path.join(MODELS_DIRPATH, "kmeans_means.pkl"), "wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

import os
import pickle
from datetime import datetime

import librosa
import numpy as np

from src.constants import N_CLUSTERS, OUTPUT_DIR, SAMPLING_RATE, MODELS_DIRPATH
from src.generator.abstract_generator import concatenate_chunks


class ClusterGenerator:
    def __init__(self):
        with open(os.path.join(
                MODELS_DIRPATH, "kmeans_cluster_to_chunks.pkl"
        ), "rb") as file:
            self.cluster_to_chunks = pickle.load(file)

    def get_waveforms(self):
        return {
            cluster: self.get_waveform(cluster)
            for cluster in self.cluster_to_chunks
        }

    def get_waveform(self, cluster):
        chunks = self.cluster_to_chunks[cluster]
        waveform = concatenate_chunks(chunks)
        return np.asarray(waveform)


def main():
    start_time = datetime.now()

    cluster_generator = ClusterGenerator()
    cluster = 0
    while cluster < min(10, N_CLUSTERS):
        waveform = cluster_generator.get_waveform(cluster)

        filepath = os.path.join(
            OUTPUT_DIR, cluster_generator.__class__.__name__,
            f"cluster_{cluster}.wav"
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        librosa.output.write_wav(filepath, waveform, SAMPLING_RATE)

        print(f"Finished processing {cluster + 1} clusters    "
              f"Elapsed time {datetime.now() - start_time}")
        cluster += 1


if __name__ == "__main__":
    main()

import os
import pickle

from src.constants import MODELS_DIRPATH
from src.generator.abstract_generator import AbstractGenerator, argmin


class KmeansNearestGenerator(AbstractGenerator):
    def __init__(self):
        super().__init__()

        with open(os.path.join(
                MODELS_DIRPATH, "kmeans_cluster_to_chunks.pkl"
        ), "rb") as file:
            self.cluster_to_chunks = pickle.load(file)

        with open(os.path.join(MODELS_DIRPATH, "rescale.pkl"), "rb") as file:
            self.chunk_to_features = pickle.load(file)

    def generate_chunk(self, template_chunk):
        template_cluster = self.chunk_to_cluster[template_chunk]
        template_features = self.chunk_to_features[template_chunk]

        def phi(item):
            features = self.chunk_to_features[item]
            return template_features.euclidean_distance(features)

        candidate_chunks = self.cluster_to_chunks[template_cluster]
        return argmin(candidate_chunks, phi)[0]

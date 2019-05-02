import os
import pickle
import random

from src.constants import MODELS_DIRPATH
from src.generator.abstract_generator import AbstractGenerator


class KmeansRandomGenerator(AbstractGenerator):
    def __init__(self):
        super().__init__()

        with open(os.path.join(
                MODELS_DIRPATH, "kmeans_cluster_to_chunks.pkl"
        ), "rb") as file:
            self.cluster_to_chunks = pickle.load(file)

    def generate_chunk(self, template_chunk):
        # replace this chunk with another chunk from the same cluster
        cluster = self.chunk_to_cluster[template_chunk]
        candidate_chunks = self.cluster_to_chunks[cluster]

        random.seed(a=1731)
        return random.sample(candidate_chunks, 1)[0]

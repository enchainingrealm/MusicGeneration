import os
import pickle
from abc import abstractmethod

from gensim.models import Word2Vec

from src.constants import MODELS_DIRPATH
from src.generator.abstract_generator import AbstractGenerator, argmin


class Word2VecAbstractGenerator(AbstractGenerator):
    def __init__(self):
        super().__init__()

        with open(os.path.join(
                MODELS_DIRPATH, "kmeans_cluster_to_chunks.pkl"
        ), "rb") as file:
            self.cluster_to_chunks = pickle.load(file)

        with open(os.path.join(MODELS_DIRPATH, "rescale.pkl"), "rb") as file:
            self.chunk_to_features = pickle.load(file)

        self.model = Word2Vec.load(os.path.join(
            MODELS_DIRPATH, "word2vec.model"
        ))

    def generate_chunk(self, template_chunk):
        result_cluster = self.get_nearest_cluster(template_chunk)
        result_chunk = self.get_nearest_chunk(template_chunk, result_cluster)
        return result_chunk

    @abstractmethod
    def get_nearest_cluster(self, template_chunk):
        raise NotImplementedError

    def get_nearest_chunk(self, template_chunk, result_cluster):
        def phi(item):
            template_features = self.chunk_to_features[template_chunk]
            candidate_features = self.chunk_to_features[item]

            return template_features.euclidean_distance(candidate_features)

        candidate_chunks = self.cluster_to_chunks[result_cluster]
        return argmin(candidate_chunks, phi)[0]

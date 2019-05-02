import numpy as np

from src.constants import N_CLUSTERS
from src.generator.abstract_generator import argmin
from src.generator.word2vec_abstract_generator import Word2VecAbstractGenerator


class Word2VecEuclideanGenerator(Word2VecAbstractGenerator):
    def get_nearest_cluster(self, template_chunk):
        template_cluster = self.chunk_to_cluster[template_chunk]

        def phi(item):
            return np.linalg.norm(
                self.model.wv[str(template_cluster)] - self.model.wv[str(item)]
            )

        candidate_clusters = {
            cluster for cluster in range(N_CLUSTERS)
            if cluster != template_cluster
        }
        return argmin(candidate_clusters, phi)[0]

import os
import pickle
from abc import ABC, abstractmethod
from math import inf

import numpy as np

from src.constants import CHUNK_SECONDS, MODELS_DIRPATH
from src.structs.chunk import Chunk


class AbstractGenerator(ABC):
    def __init__(self):
        with open(os.path.join(
                MODELS_DIRPATH, "kmeans_chunk_to_cluster.pkl"
        ), "rb") as file:
            self.chunk_to_cluster = pickle.load(file)

    def generate_waveform(self, template_dirname, template_filename):
        generated_chunks = self.generate_chunks(
            template_dirname, template_filename
        )

        waveform = concatenate_chunks(generated_chunks)
        return np.asarray(waveform), generated_chunks

    def generate_chunks(self, template_dirname, template_filename):
        template_chunks = []
        index = 0
        while True:
            template_chunk = Chunk(template_dirname, template_filename, index)
            if template_chunk not in self.chunk_to_cluster:
                break   # index out of bounds

            template_chunks.append(template_chunk)
            index += 1

        return [
            self.generate_chunk(template_chunk)
            for template_chunk in template_chunks
        ]

    @abstractmethod
    def generate_chunk(self, template_chunk):
        raise NotImplementedError


def concatenate_chunks(chunks):
    y = []
    for chunk in chunks:
        y_curr = chunk.get_waveform(CHUNK_SECONDS)
        y.extend(y_curr)

    return y


def argmin(collection, phi):
    result = None
    phi_result = inf

    for item in collection:
        phi_curr = phi(item)
        if phi_curr < phi_result:
            result = item
            phi_result = phi_curr

    return result, phi_result

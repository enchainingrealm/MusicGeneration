import multiprocessing
import os
import pickle
from copy import copy
from datetime import datetime

from gensim.models import Word2Vec

from src.constants import MODELS_DIRPATH


def main():
    start_time = datetime.now()

    chunk_to_cluster, _ = load_clusters()
    sentences = to_sentences(chunk_to_cluster)

    print("Started training word2vec.")

    model = Word2Vec(
        size=50, window=5, min_count=1, seed=1731,
        workers=multiprocessing.cpu_count()
    )

    model.build_vocab(sentences, progress_per=100)
    print(f"Vocabulary size: {len(model.wv.vocab)}")

    model.train(
        sentences, total_examples=len(sentences), epochs=model.epochs,
        compute_loss=True
    )

    print("Finished training word2vec.    "
          f"Elapsed time {datetime.now() - start_time}")

    model.save(os.path.join(MODELS_DIRPATH, "word2vec.model"))


def load_clusters():
    with open(os.path.join(
            MODELS_DIRPATH, "kmeans_chunk_to_cluster.pkl"
    ), "rb") as file:
        chunk_to_cluster = pickle.load(file)

    with open(os.path.join(
            MODELS_DIRPATH, "kmeans_cluster_to_chunks.pkl"
    ), "rb") as file:
        cluster_to_chunks = pickle.load(file)

    return chunk_to_cluster, cluster_to_chunks


def to_sentences(chunk_to_cluster):
    sentences = []

    keys = set(chunk_to_cluster.keys())
    while keys:   # while keys is not empty
        sentence = []
        next_chunk = next(iter(keys))   # get an arbitrary key

        iter_chunk = copy(next_chunk)
        iter_chunk.index = 0
        while iter_chunk in keys:
            cluster = chunk_to_cluster[iter_chunk]
            sentence.append(cluster)

            keys.remove(iter_chunk)
            iter_chunk.index += 1

        sentence = list(map(str, sentence))
        sentences.append(sentence)

    return sentences


if __name__ == "__main__":
    main()

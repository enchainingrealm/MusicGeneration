import os
import pickle
from math import sqrt

from src.constants import MODELS_DIRPATH


def main():
    with open(os.path.join(MODELS_DIRPATH, "vectorizer.pkl"), "rb") as file:
        result = pickle.load(file)

    mfcc_scalar = 1 / sqrt(20)
    chroma_scalar = 1 / sqrt(12)
    for chunk, features in result.items():
        rescale(features, mfcc_scalar, chroma_scalar)

    with open(os.path.join(MODELS_DIRPATH, "rescale.pkl"), "wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)


def rescale(features, mfcc_scalar, chroma_scalar):
    features.mel_frequency_cepstral_coefficients *= mfcc_scalar
    features.chroma_frequencies *= chroma_scalar


if __name__ == "__main__":
    main()

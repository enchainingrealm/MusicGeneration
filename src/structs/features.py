import librosa
import numpy as np


class Features:
    def __init__(self, chunk_y, sr):
        self.zero_crossing_rate = zero_crossing_rate(chunk_y)
        self.spectral_centroid = spectral_centroid(chunk_y, sr)
        self.spectral_rolloff = spectral_rolloff(chunk_y, sr)
        self.mel_frequency_cepstral_coefficients\
            = mel_frequency_cepstral_coefficients(chunk_y, sr)
        self.chroma_frequencies = chroma_frequencies(chunk_y, sr)

    def to_list(self):
        return [self.zero_crossing_rate]\
               + [self.spectral_centroid]\
               + [self.spectral_rolloff]\
               + list(self.mel_frequency_cepstral_coefficients)\
               + list(self.chroma_frequencies)

    def to_vector(self):
        return np.asarray(self.to_list())

    def to_row_array(self):
        return self.to_vector().reshape((1, -1))

    def euclidean_distance(self, other):
        return np.linalg.norm(self.to_vector() - other.to_vector())


# Models the percussiveness of the sound.
def zero_crossing_rate(chunk_y):
    indicator = librosa.zero_crossings(chunk_y, pad=False)
    return np.count_nonzero(indicator)


# Models the timbre and "brightness" of the sound.
def spectral_centroid(chunk_y, sr):
    result = librosa.feature.spectral_centroid(chunk_y, sr=sr)

    assert len(result.shape) == 2, len(result.shape)
    assert result.shape[0] == 1, result.shape[0]

    return result[0, :].mean()


def spectral_rolloff(chunk_y, sr):
    result = librosa.feature.spectral_rolloff(chunk_y, sr=sr)

    assert len(result.shape) == 2, len(result.shape)
    assert result.shape[0] == 1, result.shape[0]

    return result[0, :].mean()


# Good for modeling human voices.
def mel_frequency_cepstral_coefficients(chunk_y, sr):
    result = librosa.feature.mfcc(chunk_y, sr=sr)

    assert len(result.shape) == 2, len(result.shape)
    assert result.shape[0] == 20, result.shape[0]

    return result.mean(axis=1)


# Good for modeling music audio.
def chroma_frequencies(chunk_y, sr):
    result = librosa.feature.chroma_stft(chunk_y, sr=sr)

    assert len(result.shape) == 2, len(result.shape)
    assert result.shape[0] == 12, result.shape[0]

    return result.mean(axis=1)


# Sources:
# - https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d
# - https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8
# - https://medium.com/heuristics/audio-signal-feature-extraction-and-clustering-935319d2225

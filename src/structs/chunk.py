import os

import librosa

from src.constants import SAMPLING_RATE, INPUT_DIRPATH, EXTENSIONS


class Chunk:
    def __init__(self, dirname, filename, index):
        self.dirname = dirname
        self.filename = filename
        self.index = index

    def get_waveform(self, chunk_seconds):
        audio_fullpath = self.to_audio_fullpath()

        y, _ = librosa.load(audio_fullpath, sr=SAMPLING_RATE, mono=True)

        chunk_length = round(chunk_seconds * SAMPLING_RATE)
        lo = chunk_length * self.index
        hi = min(y.size, lo + chunk_length)

        return y[lo:hi]

    def to_audio_fullpath(self):
        for extension in EXTENSIONS:
            fullpath = os.path.join(
                INPUT_DIRPATH, self.dirname, self.filename + extension
            )

            if os.path.isfile(fullpath):
                return fullpath

        raise ValueError(f"No such audio file: {str(self)}")

    def __eq__(self, other):
        return self.dirname == other.dirname\
               and self.filename == other.filename\
               and self.index == other.index

    def __hash__(self):
        code = hash(self.dirname)
        code = code * 31 + hash(self.filename)
        code = code * 31 + hash(self.index)
        return code

    def __str__(self):
        return super().__str__() + ": " + str(self.__dict__)

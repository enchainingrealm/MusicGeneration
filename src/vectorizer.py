import os
import pickle
from datetime import datetime

import librosa

from src.constants import INPUT_DIRPATH, EXTENSIONS, CHUNK_SECONDS, \
    MODELS_DIRPATH
from src.structs.chunk import Chunk
from src.structs.features import Features


def main():
    start_time = datetime.now()

    result = {}

    audio_filepaths = get_audio_filepaths()
    for count, audio_filepath in enumerate(audio_filepaths):
        chunks_y, sr = load_and_split(audio_filepath)
        for index, chunk_y in enumerate(chunks_y):
            dirname = get_dirname(audio_filepath)
            filename = get_filename(audio_filepath)

            chunk = Chunk(dirname, filename, index)
            features = Features(chunk_y, sr)
            result[chunk] = features

        print(f"Processed {count + 1} files    "
              f"Elapsed time {datetime.now() - start_time}")
        count += 1

    with open(os.path.join(MODELS_DIRPATH, "vectorizer.pkl"), "wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)


def get_audio_filepaths():
    result = []
    for sub_dirname in os.listdir(INPUT_DIRPATH):
        sub_dirpath = os.path.join(INPUT_DIRPATH, sub_dirname)
        if not os.path.isdir(sub_dirpath):
            continue

        for filename in os.listdir(sub_dirpath):
            if not is_audio_file(filename):
                continue

            filepath = os.path.join(sub_dirpath, filename)
            result.append(filepath)
    return result


def is_audio_file(filename):
    return any(
        filename.lower().endswith(extension)
        for extension in EXTENSIONS
    )


def load_and_split(audio_filepath):
    y, sr = librosa.load(audio_filepath, sr=None, mono=True)  # mono=True is
    # necessary for feature extractors to work

    chunks_y = []
    lo = 0
    while lo < y.size:
        chunk_length = round(CHUNK_SECONDS * sr)
        hi = min(y.size, lo + chunk_length)  # prevent index out of bounds

        chunks_y.append(y[lo:hi])
        lo = hi
    return chunks_y, sr


def get_dirname(audio_filepath):
    breadcrumbs = audio_filepath.split(os.sep)
    return breadcrumbs[-2]


def get_filename(audio_filepath):
    breadcrumbs = audio_filepath.split(os.sep)

    filename = breadcrumbs[-1]
    return os.path.splitext(filename)[0]  # remove the extension


if __name__ == "__main__":
    main()

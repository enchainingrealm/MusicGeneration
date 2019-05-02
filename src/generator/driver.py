import csv
import os
from datetime import datetime

import librosa

from src.constants import TEMPLATE_COMPOSITIONS, OUTPUT_DIR, SAMPLING_RATE
from src.generator.kmeans_nearest_generator import KmeansNearestGenerator
from src.generator.kmeans_random_generator import KmeansRandomGenerator
from src.generator.word2vec_cosine_generator import Word2VecCosineGenerator
from src.generator.word2vec_euclidean_generator import \
    Word2VecEuclideanGenerator


def main():
    start_time = datetime.now()

    generators = {
        KmeansRandomGenerator(),
        KmeansNearestGenerator(),
        Word2VecCosineGenerator(),
        Word2VecEuclideanGenerator()
    }

    count = 0
    for generator in generators:
        for template_dirname, template_filename in TEMPLATE_COMPOSITIONS:
            template_filename, extension = os.path.splitext(template_filename)

            # Generate a new composition.
            waveform, chunks = generator.generate_waveform(
                template_dirname, template_filename
            )

            # Get the filename to save it at.
            filepath = os.path.join(
                OUTPUT_DIR, generator.__class__.__name__,
                template_dirname + " - " + template_filename
            )
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save the waveform.
            librosa.output.write_wav(filepath + ".wav", waveform, SAMPLING_RATE)

            # Save the chunks.
            with open(filepath + ".csv", "w") as file:
                csv_writer = csv.writer(file, delimiter=",")
                csv_writer.writerow(["dirname", "filename", "index"])

                for chunk in chunks:
                    csv_writer.writerow([
                        chunk.dirname, chunk.filename, chunk.index
                    ])

            # Display progress.
            print(f"Finished processing {count + 1} waveforms    "
                  f"Elapsed time {datetime.now() - start_time}")
            count += 1


if __name__ == "__main__":
    main()

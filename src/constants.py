from root import from_root

# -------------------------
# Globally useful constants

MODELS_DIRPATH = from_root("./data/models/")

# ------------------------------------------------------------------------------
# Vectorizer related constants

# Where to look for training data.
INPUT_DIRPATH = from_root("./data/input/")
EXTENSIONS = {".wav", ".flac", ".mp3"}

# Size of chunks to split a loaded audio file into.
CHUNK_SECONDS = 1.0

# ------------------------------------------------------------------------------
# K-Means related constants

N_CLUSTERS = 1000

# ------------------------------------------------------------------------------
# Generator related constants

OUTPUT_DIR = from_root("./data/output/")

TEMPLATE_COMPOSITIONS = {
    ("Beethoven - Piano Sonatas - Goode",
     "Piano Sonata No. 21 Op. 53 'Waldstein' in C+ Mvt. 1.wav"),
    ("Clementi - Piano Sonatas, Sonatinas, Piano Works - Shelley",
     "Piano Sonata Op. 25 No. 5 in F#- Mvt. 3.flac"),
    ("Haydn - Piano Sonatas, Piano Works - McCabe",
     "Piano Sonata No. 60 Hob. XVI;50 in C+ Mvt. 1.wav"),
    ("Hummel - Piano Sonatas - Hobson",
     "Piano Sonata No. 5 Op. 81 in F#- Mvt. 1.wav"),
    ("Mozart - Piano Sonatas, 2 Pianos, Piano 4-Hands - Haebler",
     "Piano Sonata KV 448 (375a) in D+ Mvt. 1.mp3"),
    ("Schubert - Piano Sonatas - Kempff",
     "Piano Sonata No. 21 D. 960 in Bb+ Mvt. 1.wav")
}

# Sampling rate used when concatenating generated chunks together.
SAMPLING_RATE = 44100

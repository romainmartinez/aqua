import numpy as np

random_seed = 42
np.random.seed(random_seed)

# metrics -------------------
available_targets = [
    "eb max force",
    "eb mean force",
    "eb sd force",
    "eb max height",
    "eb min height",
    "eb mean height",
    "eb sd height",
    "eb max-min height",
    "bb",
]
default_targets = ["bb", "eb mean height", "eb mean force"]

normalization_strategies = ["None", "Weight", "Weight x Height", "IMC"]

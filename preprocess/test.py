import numpy as np

processed_folder = "UCI_HAR/processed_data"

labels = np.load(f"{processed_folder}/labels.npy")

print(np.unique(labels, return_counts = True))
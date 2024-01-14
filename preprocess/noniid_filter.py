import numpy as np

processed_folder = "UCI_HAR/processed_data"
filtered_folder = "UCI_HAR/filtered_data"

labels = np.load(f"{processed_folder}/labels.npy")
acc_features = np.load(f"{processed_folder}/acc_features.npy")
gyro_features = np.load(f"{processed_folder}/gyro_features.npy")

# Find indices of labels 0, 1, 2
indices_to_delete = np.where((labels >= 0) & (labels <= 2))[0]

# Randomly choose 1/4 of the indices to keep
selected_indices = np.random.choice(
    indices_to_delete, size=3 * len(indices_to_delete) // 4, replace=False
)

# Delete corresponding entries
labels_filtered = np.delete(labels, selected_indices, axis=0)
acc_features_filtered = np.delete(acc_features, selected_indices, axis=0)
gyro_features_filtered = np.delete(gyro_features, selected_indices, axis=0)


# Save the filtered data
np.save(f"{filtered_folder}/labels.npy", labels_filtered)
np.save(f"{filtered_folder}/acc_features.npy", acc_features_filtered)
np.save(f"{filtered_folder}/gyro_features.npy", gyro_features_filtered)

unique_values, count = np.unique(labels_filtered, return_counts=True)
print(unique_values)
print(count)

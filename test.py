import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the dataset
data = pd.read_csv("preprocessed_diabetes_dataset.csv")
data = data[:50]

# Remove any non-numeric columns
numeric_data = data.drop(['gender_Female', 'gender_Male'], axis=1)

# Perform hierarchical clustering
linked = linkage(numeric_data, 'single')

# Plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

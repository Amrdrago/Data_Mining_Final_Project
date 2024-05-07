import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

data = pd.read_csv('preprocessed_diabetes_dataset.csv')

# Take a random sample (e.g., 20% of the data)
sample_size = int(0.2 * len(data))  # Adjust the sample size as needed
data_sample = data.sample(n=sample_size, random_state=2)

# K-Medoids clustering
kmedoids = KMedoids(n_clusters=15, random_state=70)
cluster_labels = kmedoids.fit_predict(data_sample)
cluster_labels += 1
# Add cluster labels to the sample dataframe
data_sample['Cluster'] = cluster_labels

# Get the indices of the medoids
medoid_indices = kmedoids.medoid_indices_

# Print centroids of each cluster
print("Centroids of each cluster:")
for cluster_num, medoid_index in enumerate(medoid_indices):
    centroid_row_num = data_sample.index[medoid_index] + 2  # Increment by 2 instead of 1
    centroid = data_sample.iloc[medoid_index]
    print(f"Cluster {cluster_num+1} centroid (Row {centroid_row_num}):")
    pd.options.display.float_format = '{:.10f}'.format  # Set to display all decimal places
    print(centroid)

data_sample.to_csv('k-medoid_algo_sample.csv', index=False)

# Calculate silhouette score
silhouette_avg = silhouette_score(data_sample, cluster_labels)
print("Silhouette Score:", silhouette_avg)

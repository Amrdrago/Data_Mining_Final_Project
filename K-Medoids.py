import pandas as pd
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

# Read data
data = pd.read_csv('preprocessed_diabetes_dataset.csv')

# Take a random sample (e.g., 20% of the data)
sample_size = int(0.1 * len(data))  # Adjust the sample size as needed
data_sample = data.sample(n=sample_size, random_state=2)

# K-Medoids clustering with 3 clusters
kmedoids = KMedoids(n_clusters=2, random_state=2)
cluster_labels = kmedoids.fit_predict(data_sample)
cluster_labels += 1
# Add cluster labels to the sample dataframe
data_sample['Cluster'] = cluster_labels

# Get the indices of the medoids
medoid_indices = kmedoids.medoid_indices_

# Plotting all clusters together
plt.figure(figsize=(10, 8))

# Plot each cluster
for cluster_num, medoid_index in enumerate(medoid_indices):
    cluster_data = data_sample[data_sample['Cluster'] == (cluster_num + 1)]
    plt.scatter(cluster_data['HbA1c_level'], cluster_data['blood_glucose_level'], label=f'Cluster {cluster_num + 1}')

# Plot medoids
medoids = data_sample.iloc[medoid_indices]
plt.scatter(medoids['HbA1c_level'], medoids['blood_glucose_level'], c='black', marker='X', s=100, label='Medoids')

plt.xlabel('HbA1c Level')
plt.ylabel('Blood Glucose Level')
plt.title('K-Medoids Clustering')
plt.legend()
plt.grid(True)
plt.show()


# Calculate silhouette score
silhouette_avg = silhouette_score(data_sample[['HbA1c_level', 'blood_glucose_level']], cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

for cluster_num, medoid_index in enumerate(medoid_indices):
    centroid = data_sample.iloc[medoid_index]
    row_number = centroid.name  # Getting the index of the row
    print(f"Centroid for Cluster {cluster_num + 1} (Row {row_number+2}):")
    print(f"HbA1c Level: {centroid['HbA1c_level']}, Blood Glucose Level: {centroid['blood_glucose_level']}")
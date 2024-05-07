import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Read CSV file
data = pd.read_csv('preprocessed_diabetes_dataset.csv')

# Take a random sample (e.g., 20% of the data)
sample_size = int(0.2 * len(data))  # Adjust the sample size as needed
data_sample = data.sample(n=sample_size, random_state=0)

# Perform PCA
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_sample)

# K-Medoids clustering
kmedoids = KMedoids(n_clusters=5, random_state=0)
cluster_labels = kmedoids.fit_predict(data_sample)
cluster_labels += 1
# Add cluster labels to the sample dataframe
data_sample['Cluster'] = cluster_labels

# Write Results to CSV
data_sample.to_csv('k-medoid_algo_sample.csv', index=False)

# Calculate silhouette score
silhouette_avg = silhouette_score(data_sample, cluster_labels)
print("Silhouette Score:", silhouette_avg)


# Plot 3D PCA
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each cluster
for cluster in range(1, 6):
    cluster_data = data_pca[data_sample['Cluster'] == cluster]
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], label=f'Cluster {cluster}')

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.set_title('3D PCA Plot with K-Medoids Clustering')
plt.legend()
plt.show()

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

data = pd.read_csv('preprocessed_diabetes_dataset.csv')

# Take a random sample (e.g., 20% of the data)
sample_size = int(0.2 * len(data))  # Adjust the sample size as needed
data_sample = data.sample(n=sample_size, random_state=2)

n_clusters =15

# Hierarchical clustering
agg_clustering = AgglomerativeClustering(n_clusters)
cluster_labels = agg_clustering.fit_predict(data_sample)
cluster_labels += 1
# Add cluster labels to the sample dataframe
data_sample['Cluster'] = cluster_labels

# Sort cluster counts by index (cluster number)
cluster_counts_sorted = data_sample['Cluster'].value_counts().sort_index()

print("Centroids of each cluster:")
for cluster_num in range(1, n_clusters+1):
    centroid = data_sample[data_sample['Cluster'] == cluster_num].mean()
    print(f"Cluster {cluster_num} centroid:")
    pd.options.display.float_format = '{:.10f}'.format  # Set to display all decimal places
    print(centroid)
    print("--------------------------------------------")

data_sample.to_csv('hierarchical_clustering_algo_sample.csv', index=False)

# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(data_sample, cluster_labels)
print("\nDavies-Bouldin Index:", db_index)

# Calculate silhouette score
silhouette_avg = silhouette_score(data_sample, cluster_labels)
print("\nSilhouette Score:", silhouette_avg)


# Here's how it's calculated for a single data point:

# a(i): The average distance between the data point and all other points in the same cluster. This represents cohesion.

# b(i): The smallest average distance between the data point and all points in any other cluster, of which the data point is not a member. This represents separation.

# s(i): The silhouette score for the data point is then given by (b(i) - a(i)) / max(a(i), b(i)).



# Calculate Cluster Dispersion: Measure the average distance of each point in the cluster to the centroid or using the variance within the cluster.

# Calculate Cluster Separation: his is usually done by computing a distance or dissimilarity metric between the centroids of the two clusters.

# Compute the Davies-Bouldin Index: Finally, compute the Davies-Bouldin Index using the formula:DB = (1 / n) * Σ(max(R_ij + R_ji))

# R_ij is the average distance between cluster i and cluster j.

# The maximum value of (R_ij + R_ji) is chosen for each cluster i.
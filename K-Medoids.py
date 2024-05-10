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

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting each cluster
for cluster_num, medoid_index in enumerate(medoid_indices):
    cluster_data = data_sample[data_sample['Cluster'] == (cluster_num + 1)]
    ax.scatter(cluster_data['HbA1c_level'], cluster_data['blood_glucose_level'], cluster_data['diabetes'], label=f'Cluster {cluster_num + 1}')

# Plotting medoids
medoids = data_sample.iloc[medoid_indices]
ax.scatter(medoids['HbA1c_level'], medoids['blood_glucose_level'], medoids['diabetes'], c='black', marker='X', s=100, label='Medoids')

ax.set_xlabel('HbA1c Level')
ax.set_ylabel('Blood Glucose Level')
ax.set_zlabel('Diabetes')
ax.set_title('K-Medoids Clustering in 3D')
plt.legend()
plt.show()

# 2D plot between all 3 features
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Feature combinations
feature_combinations = [('HbA1c_level', 'blood_glucose_level'),
                        ('HbA1c_level', 'diabetes'),
                        ('blood_glucose_level', 'diabetes')]

for i, (feat1, feat2) in enumerate(feature_combinations):
    ax = axes[i]
    for cluster_num, medoid_index in enumerate(medoid_indices):
        cluster_data = data_sample[data_sample['Cluster'] == (cluster_num + 1)]
        ax.scatter(cluster_data[feat1], cluster_data[feat2], label=f'Cluster {cluster_num + 1}')

    ax.scatter(medoids[feat1], medoids[feat2], c='black', marker='X', s=100, label='Medoids')
    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    ax.set_title(f'{feat1} vs {feat2}')
    ax.legend()

plt.tight_layout()
plt.show()

# Save sample with cluster labels
data_sample.to_csv('k-medoid_algo_sample.csv', index=False)

# Calculate silhouette score
silhouette_avg = silhouette_score(data_sample, cluster_labels)
print("\n\nSilhouette Score:", silhouette_avg)

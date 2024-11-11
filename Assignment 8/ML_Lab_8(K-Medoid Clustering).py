import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from pyclustering.cluster.kmedoids import kmedoids

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Convert the data to a list for pyclustering (list of lists)
data = iris_df.values.tolist()

# Initializing the K-Medoids algorithm with k=3 (since the Iris dataset has 3 species)
initial_medoids = [0, 50, 100]  # Just an example, you can pick other points too
kmedoids_instance = kmedoids(data, initial_medoids)

# Run the clustering
kmedoids_instance.process()

# Get the resulting clusters and the medoids
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()

# Plotting the clusters
pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris_df)  # Exclude the 'cluster' column for PCA
iris_df['pca1'] = iris_pca[:, 0]
iris_df['pca2'] = iris_pca[:, 1]

# Create a new column for cluster labels based on the kmedoids results
iris_df['cluster'] = -1
for cluster_idx, cluster in enumerate(clusters):
    for point_idx in cluster:
        iris_df.loc[point_idx, 'cluster'] = cluster_idx

# Plotting the clusters with medoids
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_df, x='pca1', y='pca2', hue='cluster', palette='viridis', s=100, alpha=0.7)

# Plot the medoids (centroids) for each cluster
medoid_points = iris_df.iloc[medoids]
plt.scatter(medoid_points['pca1'], medoid_points['pca2'], c='red', s=200, marker='X', label='Medoids')

# Enhancing the plot
plt.title('K-Medoids Clustering on Iris Dataset (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# Print out the medoids' data points and their corresponding cluster
print("Medoids' data points and their corresponding clusters:")
print(medoid_points)

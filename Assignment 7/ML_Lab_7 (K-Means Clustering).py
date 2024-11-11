import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Initialize the K-Means model with k=3 (since the Iris dataset has 3 classes)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(iris_df)

# Add the cluster labels to the dataset
iris_df['cluster'] = kmeans.labels_

# Use PCA to reduce dimensions for better visualization (2D plot)
pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris_df.iloc[:, :-1])  # Exclude the 'cluster' column
iris_df['pca1'] = iris_pca[:, 0]
iris_df['pca2'] = iris_pca[:, 1]

# Plotting the clusters with centroids
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_df, x='pca1', y='pca2', hue='cluster', palette='viridis', s=100, alpha=0.7)

# Transform KMeans cluster centers to PCA space
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')

# Enhancing the plot
plt.title('K-Means Clustering on Iris Dataset (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

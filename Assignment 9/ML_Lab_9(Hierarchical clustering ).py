import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

# Load the Wine dataset
wine = load_wine()
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

# Perform Hierarchical Clustering using AgglomerativeClustering
agg_clust = AgglomerativeClustering(n_clusters=3, linkage='ward')  # Using 3 clusters
wine_df['cluster'] = agg_clust.fit_predict(wine_df)

# Perform PCA to reduce to 2D for better visualization
pca = PCA(n_components=2)
wine_pca = pca.fit_transform(wine_df.drop('cluster', axis=1))  # Perform PCA to reduce to 2D

wine_df['pca1'] = wine_pca[:, 0]
wine_df['pca2'] = wine_pca[:, 1]

# Plotting the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=wine_df, x='pca1', y='pca2', hue='cluster', palette='viridis', s=100, alpha=0.7)

# Enhancing the plot
plt.title('Hierarchical Clustering on Wine Dataset (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# Create the linkage matrix for the dendrogram
Z = linkage(wine_df.drop('cluster', axis=1), 'ward')

# Plot the Dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

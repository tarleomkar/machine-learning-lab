Hierarchical Clustering on the Wine Dataset
This project demonstrates the application of Hierarchical Clustering on the Wine dataset using AgglomerativeClustering from scikit-learn. The goal is to cluster wines into distinct groups and visualize the clusters after performing PCA (Principal Component Analysis) for dimensionality reduction.

Libraries Used:
pandas: For data manipulation and creation of DataFrame.
matplotlib & seaborn: For plotting and visualization.
sklearn.datasets: To load the Wine dataset.
sklearn.cluster: For performing the hierarchical clustering using AgglomerativeClustering.
scipy.cluster.hierarchy: For generating the dendrogram and linkage matrix.
sklearn.decomposition: For dimensionality reduction using PCA.
Steps Involved:
Load the Wine Dataset:

The dataset contains 178 samples, each with 13 features, which represent different chemical properties of wines.
We load the dataset using load_wine() from sklearn.datasets and convert it into a DataFrame for easier manipulation.
wine = load_wine()
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
Perform Hierarchical Clustering:

We apply Agglomerative Clustering from scikit-learn, specifying the number of clusters as 3 (as the dataset has 3 different classes of wines).
The Ward linkage method is used, which minimizes the variance within each cluster.
The resulting cluster labels are added to the DataFrame.
agg_clust = AgglomerativeClustering(n_clusters=3, linkage='ward')
wine_df['cluster'] = agg_clust.fit_predict(wine_df)
Dimensionality Reduction with PCA:

Since the dataset has 13 features, it is hard to visualize in 2D. We use PCA to reduce the data to 2 dimensions for better visualization.
The first and second principal components are added to the DataFrame as pca1 and pca2.
pca = PCA(n_components=2)
wine_pca = pca.fit_transform(wine_df.drop('cluster', axis=1))
wine_df['pca1'] = wine_pca[:, 0]
wine_df['pca2'] = wine_pca[:, 1]
Visualizing the Clusters:

We plot the 2D representation of the data points using PCA, where each data point is colored based on its assigned cluster.
Seaborn's scatterplot is used to visualize how the wines are grouped after clustering.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=wine_df, x='pca1', y='pca2', hue='cluster', palette='viridis', s=100, alpha=0.7)
plt.title('Hierarchical Clustering on Wine Dataset (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()
Dendrogram Plot:

A dendrogram is plotted using the linkage matrix, which visually represents the hierarchical structure of the clusters.
The dendrogram helps to determine how clusters are merged and at what distances.
Z = linkage(wine_df.drop('cluster', axis=1), 'ward')
plt.figure(figsize=(12, 8))
dendrogram(Z)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
Results:
Scatter Plot: Shows the 2D visualization of the clustered data after PCA reduction, with different colors representing different clusters.
Dendrogram: Displays the hierarchical clustering process, where the y-axis shows the distance (or dissimilarity) between merged clusters.
Conclusion:
The hierarchical clustering technique successfully groups the Wine dataset into 3 clusters based on the chemical properties of the wines. The dendrogram gives a clear view of the hierarchical merging process, which is an essential aspect of hierarchical clustering.
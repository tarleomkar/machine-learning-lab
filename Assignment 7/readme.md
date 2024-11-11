1. Loading the Dataset
python
Copy code
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
Here, you load the famous Iris dataset, which contains 150 samples of flowers with 4 features (sepal length, sepal width, petal length, petal width).
The dataset is converted into a DataFrame for easier manipulation.
2. Initializing and Fitting the K-Means Model
from sklearn.cluster import KMeans

# Initialize the K-Means model with k=3 (since the Iris dataset has 3 classes)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(iris_df)
K-Means is initialized with n_clusters=3 because we expect 3 clusters (as the Iris dataset contains 3 types of flowers: Setosa, Versicolor, and Virginica).
The fit method computes the cluster centers and assigns labels to the data points.
3. Assigning Cluster Labels
# Add the cluster labels to the dataset
iris_df['cluster'] = kmeans.labels_
The predicted cluster labels from K-Means (kmeans.labels_) are added as a new column called 'cluster' to the DataFrame.
4. Dimensionality Reduction using PCA
from sklearn.decomposition import PCA

# Use PCA to reduce dimensions for better visualization (2D plot)
pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris_df.iloc[:, :-1])  # Exclude the 'cluster' column
iris_df['pca1'] = iris_pca[:, 0]
iris_df['pca2'] = iris_pca[:, 1]
PCA (Principal Component Analysis) is used to reduce the data from 4 dimensions to 2 dimensions for visualization.
n_components=2 specifies that we want to reduce it to 2 principal components (pca1 and pca2).
These components are added as new columns to the DataFrame.
5. Plotting the Clusters and Centroids
import matplotlib.pyplot as plt
import seaborn as sns

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
This part uses Seaborn and Matplotlib to visualize the clusters.
Each point represents a flower sample, colored according to its assigned cluster (0, 1, or 2).
Centroids (cluster centers) are transformed to the PCA space and plotted as red 'X' markers.
The hue='cluster' parameter colors the points based on their assigned cluster label.
Understanding the Output Plot:
The 3 clusters are shown in different colors (e.g., purple, yellow, and green).
The red 'X' markers represent the centroids of each cluster.
The plot helps visualize how K-Means has grouped the data points based on the reduced 2D features.
Is This Correct?
Yes, the code and the output you got are correct. You have successfully implemented K-Means Clustering on the Iris dataset and visualized it using PCA.

Performance Note: K-Means might not perfectly classify the original Iris classes because it is an unsupervised learning algorithm and does not use the actual labels.
Improvement Suggestion: For further analysis, you can compare the clusters with the original labels to evaluate the clustering performance.
Let me know if you need any more help or further explanation!
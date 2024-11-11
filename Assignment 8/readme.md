1. Imports:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from pyclustering.cluster.kmedoids import kmedoids
pandas: Used for data manipulation and creating a DataFrame.
matplotlib.pyplot: For plotting graphs.
seaborn: For creating visually appealing plots.
PCA (Principal Component Analysis): A dimensionality reduction technique used here to reduce the dataset to 2D for visualization.
load_iris: A function from scikit-learn to load the Iris dataset.
kmedoids: The K-Medoids clustering algorithm from the pyclustering library.
2. Loading and Preparing the Data:
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
load_iris(): Loads the Iris dataset, which contains 150 samples of iris flowers, each with 4 features (sepal length, sepal width, petal length, petal width).
iris_df: Converts the iris dataset into a Pandas DataFrame with the features as columns.
3. Data Conversion for K-Medoids:
data = iris_df.values.tolist()
Converts the DataFrame into a list of lists (data), as the pyclustering.kmedoids algorithm requires the input data in this format.
4. Initializing K-Medoids:
initial_medoids = [0, 50, 100]
kmedoids_instance = kmedoids(data, initial_medoids)
initial_medoids: Specifies the initial points (indexes 0, 50, and 100) that will serve as the starting "medoids" (central points of the clusters). These points are picked arbitrarily; in practice, you may want to use a more sophisticated method to select them.
kmedoids_instance: Initializes the K-Medoids clustering algorithm with the provided data and initial medoids.
5. Running the Clustering Algorithm:
kmedoids_instance.process()
This runs the K-Medoids algorithm on the data, where the algorithm iteratively assigns points to clusters and updates the medoids based on the current cluster assignments.
6. Retrieving Clusters and Medoids:
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()
clusters: A list of clusters, where each cluster is represented as a list of indices of the points that belong to it.
medoids: The final medoids (central points) for each cluster.
7. PCA for Dimensionality Reduction:
pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris_df)
iris_df['pca1'] = iris_pca[:, 0]
iris_df['pca2'] = iris_pca[:, 1]
PCA(n_components=2): Reduces the data from 4 dimensions to 2 dimensions, making it easier to visualize in a 2D plot.
iris_pca: Transforms the Iris dataset into 2D space.
iris_df['pca1'] and iris_df['pca2']: Stores the two PCA components (the transformed 2D values) into the DataFrame for plotting.
8. Assigning Clusters to Data:
iris_df['cluster'] = -1
for cluster_idx, cluster in enumerate(clusters):
    for point_idx in cluster:
        iris_df.loc[point_idx, 'cluster'] = cluster_idx
Adds a new column cluster to the iris_df DataFrame, initializing it to -1.
Loops through the clusters, and for each point in the cluster, assigns the corresponding cluster index to the cluster column in the DataFrame.
9. Plotting the Clusters:
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_df, x='pca1', y='pca2', hue='cluster', palette='viridis', s=100, alpha=0.7)
sns.scatterplot: Plots the data points in 2D space using the first and second PCA components (pca1 and pca2), with points colored according to their cluster label (hue='cluster').
10. Plotting the Medoids:
medoid_points = iris_df.iloc[medoids]
plt.scatter(medoid_points['pca1'], medoid_points['pca2'], c='red', s=200, marker='X', label='Medoids')
medoid_points: Extracts the data points corresponding to the medoids from the DataFrame.
plt.scatter: Plots the medoids as red 'X' markers on the 2D plot.
11. Enhancing the Plot:
plt.title('K-Medoids Clustering on Iris Dataset (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()
Sets the plot's title and axis labels.
Displays a legend to label the clusters.
plt.show(): Displays the plot.
12. Displaying the Medoids' Data Points:
print("Medoids' data points and their corresponding clusters:")
print(medoid_points)
Prints the data points corresponding to the medoids along with their cluster labels.
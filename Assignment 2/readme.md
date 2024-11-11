2. Implement Principal Component Analysis (PCA) using python.
Explanation:

1. Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
NumPy (numpy): For numerical operations.
Matplotlib (matplotlib.pyplot): For visualizing the results.
Scikit-learn (sklearn):
load_iris: Loads the Iris dataset.
PCA: For performing Principal Component Analysis.
StandardScaler: For standardizing the dataset.

2. Loading the Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target
The Iris dataset is a standard dataset used in machine learning. It contains 150 samples of 3 species of iris flowers, each described by 4 features: sepal length, sepal width, petal length, and petal width.
X: Features (shape: 150 samples, 4 features).
y: Target labels (species of iris flower).

3. Data Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Standardization is a mandatory step before applying PCA because it ensures each feature contributes equally by having:
A mean of 0.
A variance of 1.
Why Standardization?: PCA is affected by the scale of the data. Standardizing makes sure that each feature has the same importance.

4. Applying PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
n_components=0.95: This means we want to retain at least 95% of the variance in the data. PCA will automatically choose the number of components that meet this requirement.
X_pca: Transformed dataset with reduced dimensions.
Why PCA?:

PCA reduces the number of features while retaining most of the variance, making the data easier to analyze and visualize.

5. Printing Results
print(f"Original number of features: {X.shape[1]}")
print(f"Reduced number of features: {X_pca.shape[1]}")
print(f"Explained variance by each principal component: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
Explained Variance Ratio: Shows how much variance each principal component captures.
Cumulative Explained Variance: Helps us understand how much total variance is explained as we include more components.
Example Output:
Original number of features: 4
Reduced number of features: 2
Explained variance by each principal component: [0.72770452 0.23030523]
Cumulative explained variance: [0.72770452 0.95800975]
The original dataset had 4 features.
PCA reduced it to 2 features that together explain 95.8% of the total variance.

6. Visualization of PCA Results
if X_pca.shape[1] == 2:
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=150)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA on Iris Dataset (2D)')
    plt.colorbar(label='Species')
    plt.show()
If the reduced data has 2 components, it is visualized in a 2D scatter plot.
Color Coding (c=y): The points are colored based on the target species.
Plot Analysis:

The scatter plot shows the separation between the different species based on the first two principal components.

7. 3D Visualization (Optional)
elif X_pca.shape[1] == 3:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', edgecolor='k', s=150)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.title('PCA on Iris Dataset (3D)')
    plt.show()
If PCA reduced the data to 3 components, it will be visualized in 3D.
Key Points for Practical Viva:
What is PCA?:

PCA is a technique for reducing the dimensionality of data while retaining most of the variance. It transforms the data into new features (principal components) that are uncorrelated.
Why use PCA?:

To reduce computational costs.
To remove noise and improve model performance.
To enable better visualization of high-dimensional data.
Why Standardize the Data Before PCA?:

PCA is sensitive to the scale of the data. Features with larger scales may dominate the analysis, leading to biased results.
How is the Number of Components Chosen?:

Using n_components=0.95 means we choose enough principal components to explain at least 95% of the variance.
Interpretation of Explained Variance:

The explained variance ratio shows how much information (variance) is captured by each principal component.
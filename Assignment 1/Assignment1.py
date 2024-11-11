# a) Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# b) Importing the Dataset
# Manually creating the DataFrame based on the provided data
data = {
    'Region': ['India', 'Brazil', 'USA', 'Brazil', 'USA', 'India', 'Brazil', 'India', 'USA', 'India'],
    'Age': [49, 32, 35, 43, 45, 40, np.nan, 53, 55, 42],
    'Income': [86400, 57600, 64800, 73200, np.nan, 69600, 62400, 94800, 99600, 80400],
    'Online Shopper': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# c) Handling Missing Data
# Filling missing values for 'Age' and 'Income' columns using mean values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Income'].fillna(df['Income'].mean(), inplace=True)

# d) Handling of Categorical Data
# Encoding categorical data for 'Region' and 'Online Shopper'
label_encoder_region = LabelEncoder()
label_encoder_online_shopper = LabelEncoder()

df['Region'] = label_encoder_region.fit_transform(df['Region'])
df['Online Shopper'] = label_encoder_online_shopper.fit_transform(df['Online Shopper'])

# Display the processed DataFrame
print("\nProcessed DataFrame:")
print(df)

# e) Splitting the dataset into training and testing datasets
X = df[['Region', 'Age', 'Income']]
y = df['Online Shopper']

# Splitting with 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# f) Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Performing PCA
pca = PCA(n_components=2)  # Reducing to 2 components for visualization
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)
print("\nTraining data after PCA:")
print(X_train_pca)
print("\nTesting data after PCA:")
print(X_test_pca)

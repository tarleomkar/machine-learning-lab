import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Set the working directory (update the path to your directory)
os.chdir(r"C:\Users\HP\Desktop\MCA\SYMCA\2 ML\Practical Lab\Running\Assignment 4\Emails.csv")

# Attempt to load the dataset
file_path = 'emails.csv'

# Check if the file exists
if not os.path.isfile(file_path):
    print(f"The file {file_path} was not found in the current directory.")
    print("Files in the directory:")
    print(os.listdir(os.getcwd()))
    raise FileNotFoundError(f"The file {file_path} was not found in the current directory.")

# Load the dataset and handle missing values
df = pd.read_csv(file_path, na_values=["??", "###"])

# Display the first few rows and columns of the dataset to understand its structure
print("Original Dataset:\n")
print(df.head())
print("\nColumn Names:\n")
print(df.columns)

# Check for missing values in the dataset
print("\nMissing Values in the Dataset:\n")
print(df.isna().sum())

# Separate features and target variable
# Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
# Identify categorical columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

# Handle missing values in numeric columns using mean imputation
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# Handle missing values in categorical columns using the most frequent value imputation
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Confirm that there are no missing values left
print("\nMissing Values After Imputation:\n")
print(df.isna().sum())

# Convert categorical features to numeric using Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Identify the target column name
target_column = 'target'  # Replace this with the actual name of your target column

# Verify that the target column exists in the DataFrame
if target_column not in df.columns:
    raise KeyError(f"The target column '{target_column}' was not found in the dataset.")

# Separate features and target variable
x = df.drop(columns=[target_column]).values  # Independent variables (features)
y = df[target_column].values  # Dependent variable (target)

# Check the distribution of the target labels to ensure there is more than one class
print("\nTarget Label Distribution (Before Encoding):")
print(df[target_column].value_counts())

# Convert target variable to binary classification
# Example: 'spam' to 1, 'ham' (non-spam) to 0
y_binary = np.where(y == 'spam', 1, 0)

# Check the distribution after encoding to ensure binary values (0 and 1)
print("\nTarget Label Distribution (After Encoding):")
print(pd.Series(y_binary).value_counts())

# Split the data into training and testing sets
xtr, xt, ytr, yt = train_test_split(x, y_binary, test_size=0.2, random_state=90)

# Standardize the features
sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xt = sc.transform(xt)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(xtr, ytr)

# Predict on the testing set
y_pred = knn.predict(xt)

# Evaluate and print model performance
print("\nAccuracy Score:")
print(accuracy_score(yt, y_pred))

# Generate and print the confusion matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(yt, y_pred)
print(conf_matrix)

# Ensure the confusion matrix is 2x2 (for binary classification)
if conf_matrix.shape == (2, 2):
    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    TN = conf_matrix[0, 0]
    FN = conf_matrix[1, 0]
    
    print(f"\nTrue Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")

    # Calculate and display additional performance metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
else:
    print("\nConfusion Matrix is not 2x2, which suggests there might be an issue with class labels or predictions.")
    print("Confusion Matrix shape:", conf_matrix.shape)

# Implement Naïve Bayes Classifier on Data set
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load dataset
dataset = pd.read_csv(r"C:\Users\HP\Desktop\MCA\SYMCA\2 ML\Practical Lab\Running\Assignment 5\Emails.csv")

# Check column names to ensure correct one
print("Columns in dataset: ", dataset.columns)

# Replace the 'Category' values (assuming it's 0 for spam and 1 for ham)
dataset['Category'] = dataset['Category'].replace(0, 'spam')
dataset['Category'] = dataset['Category'].replace(1, 'ham')

# Print the dataset
print("Dataset: \n")
print(dataset)

# Define feature variables (X) and target variable (y)
x = dataset.iloc[:, 1:-1].values  # Assuming the messages are in columns 1 to second-last column
y = dataset.iloc[:, -1].values  # The target is the last column ('Category')

# Split dataset into training and testing sets (80% for training, 20% for testing)
xtr, xt, ytr, yt = train_test_split(x, y, test_size=0.2, random_state=90)

# Print the split datasets
print("\nTraining Dataset Independent Variables:")
print(xtr)
print("\nTraining Dataset Dependent Variable:")
print(ytr)
print("\nTesting Dataset Independent Variables:")
print(xt)
print("\nTesting Dataset Dependent Variable:")
print(yt)

# Standardize the data (to ensure features have a mean of 0 and standard deviation of 1)
sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xt = sc.transform(xt)

# Print the standardized training set
print("\nAfter Standardizing the Dataset:")
print(xtr)

# Initialize the Support Vector Classifier (SVM) model
svm = SVC(kernel='linear', random_state=0)
svm.fit(xtr, ytr)  # Train the model on the training data

# Make predictions on the test set
y_pred = svm.predict(xt)

# Print the classification report
print("\n\n\t\t\tClassification Report (SVM)")
print(classification_report(yt, y_pred))

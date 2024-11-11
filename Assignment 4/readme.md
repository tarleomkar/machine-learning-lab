Code Explanation

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
Explanation:

pandas is used to handle and manipulate the dataset.
os is used to interact with the operating system, such as checking if the file exists.
numpy provides support for mathematical operations, especially arrays.
train_test_split is used to split the dataset into training and testing sets.
GridSearchCV is used to tune hyperparameters of a model using cross-validation.
StandardScaler is used for feature scaling (standardizing data).
LabelEncoder is used to convert categorical labels (spam/ham) into numerical labels.
KNeighborsClassifier is the classification model (KNN) used to predict spam/ham.
accuracy_score, precision_score, and classification_report are used for model evaluation.

Step 1: Set Dataset Path and Check File Existence
# File path for the dataset
file_path = r'/Emails.csv'

# Check if the file exists
if os.path.exists(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
Explanation:

file_path: Here, you're specifying the path where the Emails.csv file is stored.
os.path.exists(file_path): This checks if the CSV file exists at the specified location.
If the file is found, the dataset is read into a DataFrame using pd.read_csv().
Step 2: Dataset Exploration
    # Explore the dataset
    print("Dataset Head:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
Explanation:

df.head(): Prints the first five rows of the dataset to give you a quick overview of the data.
df.info(): Displays basic information about the dataset such as the number of rows, columns, and data types of each column.
Step 3: Check for Missing Values
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
Explanation:

df.isnull().sum(): This checks if there are any missing values in the dataset by summing up the True values (which indicate missing data) in each column.
Step 4: Handle Missing Data
    # Handle missing values using forward fill
    df.ffill(inplace=True)
Explanation:

df.ffill(inplace=True): The forward fill method is used here to fill missing values. This means that if there is a missing value, it will be filled with the previous valid value in the column.
Step 5: Check for 'label' Column
    # Check for 'label' column
    if 'label' not in df.columns:
        print("Error: 'label' column not found in the DataFrame.")
        print("Available columns are:", df.columns.tolist())
Explanation:

df.columns: This checks the column names in the dataset.
If a 'label' column is not found, an error message is printed, and it lists the available columns.
Step 6: Encode the 'label' Column
    else:
        # Encode the label
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])
Explanation:

If the 'label' column is present, LabelEncoder is used to convert the categorical labels (spam/ham) into numeric labels (0 for ham, 1 for spam).
fit_transform() method is used to fit the encoder and transform the labels at the same time.
Step 7: Feature and Label Split
        # Ensure 'label' column is the target and the rest are features
        X = df.drop('label', axis=1)
        y = df['label']
Explanation:

X: This represents the feature set. The 'label' column is dropped as it's the target variable.
y: This represents the target variable (label) which is the 'label' column.
Step 8: One-Hot Encoding of Categorical Features
        # Encode categorical features if necessary
        X = pd.get_dummies(X, drop_first=True)
Explanation:

pd.get_dummies(): This method is used to perform one-hot encoding on categorical features in the dataset. It creates binary columns for each category in the features. The drop_first=True argument ensures that the first category is dropped to avoid multicollinearity.
Step 9: Check if All Features are Numeric
        # Check if all features are numeric after encoding
        if not np.issubdtype(X.dtypes, np.number).all():
            print("Error: All features must be numeric.")
Explanation:

np.issubdtype(X.dtypes, np.number).all(): This checks if all the features in the dataset are numeric after one-hot encoding. If any feature is not numeric, it raises an error.
Step 10: Split Data into Training and Test Sets
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Explanation:

train_test_split(): Splits the dataset into training and testing sets.
test_size=0.2 means 20% of the data is used for testing and 80% for training.
random_state=42 ensures that the split is reproducible.
Step 11: Feature Scaling
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
Explanation:

StandardScaler(): This is used to standardize the features by scaling them to have a mean of 0 and a standard deviation of 1.
fit_transform(): Fits the scaler to the training data and transforms it.
transform(): Only transforms the test data (using the scaler fitted on the training data).
Step 12: Hyperparameter Tuning with GridSearchCV
        # Hyperparameter tuning using GridSearchCV
        param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
Explanation:

GridSearchCV: This is used for hyperparameter tuning. It performs a grid search over specified values for hyperparameters (in this case, n_neighbors for KNN).
cv=5 means 5-fold cross-validation.
grid_search.fit(X_train, y_train): This trains the KNN model with different values of n_neighbors and selects the best one based on cross-validation.
Step 13: Model Training and Prediction
        # Best estimator
        knn = grid_search.best_estimator_

        # Make predictions
        y_pred = knn.predict(X_test)
Explanation:

grid_search.best_estimator_: This retrieves the best KNN model after tuning hyperparameters.
knn.predict(X_test): This makes predictions on the test data using the trained model.
Step 14: Model Evaluation
        # Calculate accuracy and precision
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
Explanation:

accuracy_score(): Calculates the accuracy of the model.
precision_score(): Calculates the precision of the model (how many of the predicted spams were actually spam).
Step 15: Display Evaluation Results
        print(f'\nAccuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred))
Explanation:

classification_report(): Prints the precision, recall, f1-score, and support for each class (spam and ham).
Step 16: Classify a New Email
        # Classify a new email
        new_email_features = [[10, 1, 0]]  # Replace with actual feature values
        new_email_features = scaler.transform(new_email_features)
        new_email_prediction = knn.predict(new_email_features)

        print('\nNew Email Classification:')
        print('Spam' if le.inverse_transform(new_email_prediction)[0] == 'spam' else 'Not Spam')
Explanation:

new_email_features: This represents the features of the new email (replace with actual feature values).
scaler.transform(new_email_features): Scales the new email features to match the training data's scale.
knn.predict(): Predicts whether the new email is spam or not.
le.inverse_transform(): Converts the numeric prediction back to the original label (spam/ham).
Conclusion
This code processes the email dataset, prepares the data by handling missing values, encoding labels, and scaling features, then applies KNN for classification. After training the model, it evaluates its performance, and you can use it to classify new emails as spam or not.
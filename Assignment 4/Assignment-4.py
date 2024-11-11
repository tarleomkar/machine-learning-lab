import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report

# File path for the dataset
file_path = r'C:\Users\HP\Desktop\MCA\SYMCA\2 ML\Practical Lab\Running\Assignment 4\Emails.csv'

# Check if the file exists
if os.path.exists(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Explore the dataset
    print("Dataset Head:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Handle missing values using forward fill
    df.ffill(inplace=True)

    # Update this line: Use 'Category' instead of 'label'
    if 'Category' not in df.columns:
        print("Error: 'Category' column not found in the DataFrame.")
        print("Available columns are:", df.columns.tolist())
    else:
        # Encode the label ('Category')
        le = LabelEncoder()
        df['Category'] = le.fit_transform(df['Category'])  # 'spam' will be 1 and 'ham' will be 0

        # Ensure 'Category' column is the target and the rest are features
        X = df['Message']  # Messages are features (text)
        y = df['Category'] # Target labels

        # Convert text data to numerical features using a simple encoding
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hyperparameter tuning using GridSearchCV
        param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Best estimator
        knn = grid_search.best_estimator_

        # Make predictions
        y_pred = knn.predict(X_test)

        # Calculate accuracy and precision
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        print(f'\nAccuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred))

else:
    print(f"Error: The file at {file_path} was not found.")

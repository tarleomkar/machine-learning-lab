import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import os
print("Current Working Directory:", os.getcwd())

# Load the dataset
df = pd.read_excel(r"C:\Users\shubh\Desktop\K K Wagh 1yr MCA\Second year\ML\Assignment_1_Dataset.xlsx")

print("Original Data:")
print(df.head())

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# Handle missing numerical data with mean
numerical_columns = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

# Handle missing categorical data with mode
categorical_columns = df.select_dtypes(include='object').columns
imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

# Check again for missing values after imputation
print("Missing Values After Imputation:")
print(df.isnull().sum())

# Apply OneHotEncoder to categorical columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_data = encoder.fit_transform(df[categorical_columns])
encoded_columns = encoder.get_feature_names_out(categorical_columns)

# Combine encoded categorical columns with numerical columns
df_encoded = pd.concat([df.drop(categorical_columns, axis=1), pd.DataFrame(encoded_data, columns=encoded_columns)], axis=1)

# Print the column names to check if 'Online Shopper_yes' exists
print("Columns After OneHot Encoding:")
print(df_encoded.columns)

# Save the encoded data to an Excel file
output_file = r"C:\Users\shubh\Desktop\K K Wagh 1yr MCA\Second year\ML\Updated_Assignment_1_Dataset.xlsx"
df_encoded.to_excel(output_file, index=False)

print(f"Updated data saved to: {output_file}")

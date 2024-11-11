import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_excel(r"C:\Users\shubh\Desktop\K K Wagh 1yr MCA\Second year\ML\linear_regression_dataset.xlsx")

# Display the first few rows of the dataset
print(data.head())

# Features and target
X = data[["Feature_1", "Feature_2"]]  # Features
y = data["Target"]  # Target variable

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Display the coefficients of the model
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)

# You can also display predictions vs actual values
comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(comparison.head())

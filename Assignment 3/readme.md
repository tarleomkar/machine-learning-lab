3. Implement linear regression on Data set
Explanation of the Code

1. Importing Necessary Libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
NumPy (numpy): For handling numerical operations and creating arrays.
Scikit-learn (train_test_split, LinearRegression):
train_test_split: Splits the data into training and testing sets.
LinearRegression: Creates a linear regression model.
Matplotlib (matplotlib.pyplot): For visualizing the data and the linear regression line.

2. Dataset Initialization
X = np.array([10, 9, 2, 15, 10, 16, 11, 16]).reshape(-1, 1)
y = np.array([95, 80, 10, 50, 45, 98, 38, 93])
X: Represents the number of hours spent driving (independent variable).
y: Represents the Risk Score (dependent variable).
.reshape(-1, 1): Reshapes X to have one feature column (required format for training the model).

3. Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_test_split:
Splits the dataset into 80% training and 20% testing.
random_state=42 ensures reproducibility of the results.

4. Creating and Training the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
A Linear Regression model is created and trained using the training dataset (X_train, y_train).

5. Making a Prediction
hours_spent = np.array([[20]])
predicted_risk_score = model.predict(hours_spent)
Predicts the risk score for 20 hours of driving using the trained model.

6. Displaying the Predicted Risk Score
print(f"Predicted risk score for 20 hours of driving: {predicted_risk_score[0]:.2f}")
This prints the predicted risk score for 20 hours of driving.
Example Output:
Predicted risk score for 20 hours of driving: 123.44

7. Visualizing the Data and Prediction
plt.scatter(X, y, color='blue', label='Actual data')  # Original data points
plt.plot(X, model.predict(X), color='red', label='Best fit line')  # Line of best fit
plt.scatter(hours_spent, predicted_risk_score, color='green', marker='x', s=100, label='Prediction for 20 hours')
plt.xlabel('Number of hours spent driving')
plt.ylabel('Risk Score')
plt.legend()
plt.show()
Scatter Plot: Blue points show the actual data.
Red Line: Represents the best fit line obtained from the linear regression model.
Green Marker (X): Shows the predicted risk score for 20 hours of driving.
Key Points for Practical Viva:
What is Linear Regression?:

Linear Regression is a statistical method for modeling the relationship between a dependent variable (y) and one or more independent variables (X). The goal is to fit a line (y = mx + c) that best represents the data.
Why Split Data into Training and Testing Sets?:

To evaluate the model's performance on unseen data and avoid overfitting.
Why Use reshape(-1, 1) for X?:

In Scikit-learn, the input to the model should be a 2D array (n_samples, n_features). Reshaping ensures the input has the correct shape.
How is the Prediction Done?:

After training, the model can predict new values using the learned coefficients (m and c in the line equation).
Steps to Execute the Code:
Copy and paste the entire code into a Python IDE (e.g., Jupyter Notebook, VS Code) or a Python script (.py file).
Run the code cell or script.
You should see the printed output with the predicted risk score and a plot showing the data points, the regression line, and the prediction marker.
1. Importing Required Libraries
python
Copy code
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
Explanation:

pandas: For data manipulation and analysis (reading and handling CSV files).
numpy: Provides support for numerical operations.
train_test_split: To split the dataset into training and testing sets.
classification_report: To generate a report with metrics like accuracy, precision, recall, and F1-score.
GaussianNB: Implements the Naive Bayes classifier for continuous data (used for classification).
2. Loading the Dataset
python
Copy code
try:
    print("Dataset: \n")
    dataset = pd.read_csv(r"C:\Users\HP\Desktop\MCA\SYMCA\2 ML\Practical Lab\Running\Assignment 6\Tennis.csv")  # Use raw string
except FileNotFoundError:
    print("Error: File 'Tennis.csv' not found. Please check the path.")
    exit()
Explanation:

pd.read_csv() reads the CSV file located at the given path. The r before the string indicates a raw string to handle file paths properly.
The try-except block is used for error handling. If the file is not found, it prints an error message and exits.
3. Preparing the Features and Target
python
Copy code
x = dataset[['outlook', 'temp', 'humidity', 'windy']]
y = dataset['play']
x = pd.get_dummies(x)
Explanation:

x: Contains the feature columns 'outlook', 'temp', 'humidity', and 'windy'.
y: Contains the target variable 'play'.
pd.get_dummies(): Converts categorical columns (like 'outlook') into dummy/indicator variables (one-hot encoding), making them suitable for machine learning algorithms.
4. Splitting the Data
python
Copy code
xtr, xt, ytr, yt = train_test_split(x, y, test_size=0.2, random_state=20)
Explanation:

train_test_split splits the data into training and testing sets.
test_size=0.2 means 20% of the data will be used for testing, and 80% for training.
random_state=20 ensures reproducibility of the split.
5. Fitting the Naive Bayes Classifier
python
Copy code
nbc = GaussianNB()
nbc.fit(xtr, ytr)
Explanation:

GaussianNB(): Creates an instance of the Gaussian Naive Bayes classifier.
.fit(): Trains the model using the training data (xtr and ytr).
6. Making Predictions and Evaluating the Model
python
Copy code
y_pred = nbc.predict(xt)

print("\n\n\t\t\tClassification Report")
print(classification_report(yt, y_pred))
Explanation:

nbc.predict(): Makes predictions on the test set (xt).
classification_report(): Generates a report with metrics like precision, recall, F1-score, and accuracy for evaluating the model.

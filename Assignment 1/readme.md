c:\Users\HP\Desktop\MCA\SYMCA\2 ML\Assignments\Assignment 1\Assignment1.py:21: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['Age'].fillna(df['Age'].mean(), inplace=True)
c:\Users\HP\Desktop\MCA\SYMCA\2 ML\Assignments\Assignment 1\Assignment1.py:22: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['Income'].fillna(df['Income'].mean(), inplace=True)

Processed DataFrame:
   Region        Age        Income  Online Shopper
0       1  49.000000  86400.000000               0
1       0  32.000000  57600.000000               1
2       2  35.000000  64800.000000               0
3       0  43.000000  73200.000000               0
4       2  45.000000  76533.333333               1
5       1  40.000000  69600.000000               1
6       0  43.777778  62400.000000               0
7       1  53.000000  94800.000000               1
8       2  55.000000  99600.000000               0
9       1  42.000000  80400.000000               1

Explained Variance Ratio: [0.61177743 0.35052349]

Training data after PCA:
[[-0.96956384 -0.09290664]
 [ 1.41904475  0.17099327]
 [ 2.54348702  0.31256983]
 [-2.15925389  1.23431371]
 [ 0.03701486  0.14963044]
 [ 0.02795579  1.34711286]
 [-0.14011421 -1.41284763]
 [-0.75857049 -1.70886583]]

Testing data after PCA:
[[ 2.97820372  1.75483417]
 [-2.72737865 -1.60701408]]
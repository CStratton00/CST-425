# SVM Classifier

## Load the appropriate software packages.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Import the data
colnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '50k-Prediction']
df = pd.read_csv("adult.data", names=colnames)
print(df.head(20))

# Data preprocessing
# Define each column as numerical or categorical
#df['age'] = df['age'].astype('float64')
#df['workclass'] = df['workclass'].astype('category')
#df['fnlwgt'] = df['fnlwgt'].astype('float64')
#df['education'] = df['education'].astype('category')
#df['education-num'] = df['education-num'].astype('float64')
#df['marital-status'] = df['marital-status'].astype('category')
#df['occupation'] = df['occupation'].astype('category')
#df['relationship'] = df['relationship'].astype('category')
#df['race'] = df['race'].astype('category')
#df['sex'] = df['sex'].astype('category')
#df['capital-gain'] = df['capital-gain'].astype('float64')
#df['capital-loss'] = df['capital-loss'].astype('float64')
#df['hours-per-week'] = df['hours-per-week'].astype('float64')
#df['native-country'] = df['native-country'].astype('category')

# Convert categorical variables into simple label encoders
from sklearn.preprocessing import LabelEncoder

# Creating a instance of label Encoder.
le = LabelEncoder()

# Using .fit_transform function to fit label
# encoder and return encoded label
df['workclass'] = le.fit_transform(df['workclass'])
df['education'] = le.fit_transform(df['education'])
df['marital-status'] = le.fit_transform(df['marital-status'])
df['occupation'] = le.fit_transform(df['occupation'])
df['relationship'] = le.fit_transform(df['relationship'])
df['race'] = le.fit_transform(df['race'])
df['sex'] = le.fit_transform(df['sex'])
df['native-country'] = le.fit_transform(df['native-country'])
df['50k-Prediction'] = le.fit_transform(df['50k-Prediction'])

# printing label
print(df.head(20))

## Subset the data.
x = df.drop('50k-Prediction', axis=1) # all columns except 50k Prediction column (or classifier)
y = df['50k-Prediction'] # only the 50k Prediction column (or classifier)

# Data Cleaning

## Step 1: Remove unnecessary variables

## Step 2: Remove duplicate data rows
df.drop_duplicates()

## Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Data Normalization

## Build the classification model.
print("\n--- GAUSSIAN KERNEL ---\n")
svclassifier = SVC(kernel='rbf') # Note the use of 'rbf'
svclassifier.fit(x_train, y_train)

## Run the model (make predictions).
y_pred = svclassifier.predict(x_test)

## Display classification results (quantitative and visual)

## Provide the confusion matrix for each classifier.
print("\nCONFUSION MATRIX:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

## For each classifier, compute the accuracy, sensitivity, and specificity.
accuracy = 0.2
sensitivity = 0.2
specificity = 0.2

print(f"Accuracy: {accuracy}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")

## Explain the use of the ROC curve and the meaning of the area under the ROC curve.
# TK

## Compare the results obtained with each one of the classifiers, referring to the confusion matrix and associated metrics. Are the results similar? If not, how statistically different are they? If different, what is the reason? If similar, how would you decide to use one method or another?
# TK

# References
# https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
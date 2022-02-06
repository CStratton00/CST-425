# SVM Classifier

# Load the appropriate software packages.
import numpy as np
from numpy import mean
from numpy import absolute
from numpy import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

# Import the data
colnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '50k-Prediction']
df = pd.read_csv("adult.data", names=colnames)

## Explore the data
print(df.head(20))

# Data preprocessing
## Define each column as numerical or categorical
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

## Step 1: Remove unnecessary variables


## Step 2: Remove duplicate data rows
df.drop_duplicates()

## Step 3: Convert categorical variables into numerical data using a simple label encoder
le = LabelEncoder()

### Use the .fit_transform() function to fit each column into a numerical value
df['workclass'] = le.fit_transform(df['workclass'])
df['education'] = le.fit_transform(df['education'])
df['marital-status'] = le.fit_transform(df['marital-status'])
df['occupation'] = le.fit_transform(df['occupation'])
df['relationship'] = le.fit_transform(df['relationship'])
df['race'] = le.fit_transform(df['race'])
df['sex'] = le.fit_transform(df['sex'])
df['native-country'] = le.fit_transform(df['native-country'])
df['50k-Prediction'] = le.fit_transform(df['50k-Prediction'])

## Step 4: Remove outliers
# TK

## Step 5: Print the updated data with data cleaning
print(df.head(20))

## Subset the data into x and y and lower sample amount
#x = df.drop('50k-Prediction', axis=1) # all columns except 50k Prediction column (or classifier)
#x = x.iloc[:1000,].values  # Set sample amount to first 1000 rows
#y = df['50k-Prediction'].iloc[:1000,].values # only the 50k Prediction column (or classifier), Set sample amount to first 1000 rows

# Data Normalization
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

## Build KFold Model to split and build model
print("\n--- GAUSSIAN KERNEL ---\n")
k = 5 # Do 5 folds, and split the set into 80/20
kfold = KFold(n_splits=k, random_state=1234, shuffle=True)
svclassifier = SVC(kernel='rbf') # Note the use of 'rbf'

acc_score = []

for train_index, test_index in kfold.split(x):
     x_train, x_test = x.iloc[train_index,:], x.iloc[test_index,:]
     y_train, y_test = y[train_index], y[test_index]

     svclassifier.fit(x_train,y_train)
     y_pred = svclassifier.predict(x_test)

     acc = accuracy_score(y_pred, y_test)
     acc_score.append(acc)

avg_acc_score = sum(acc_score)/k

print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))

## Split the data into training and testing sets
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


## Build the classification model.
#svclassifier = SVC(kernel='rbf') # Note the use of 'rbf'
#svclassifier.fit(x_train, y_train)

## Run the model (make predictions).
y_pred = svclassifier.predict(x_test)

## Display classification results (quantitative and visual)

## Provide the confusion matrix for each classifier.
print("\nCONFUSION MATRIX:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

## For each classifier, compute the accuracy, sensitivity, and specificity.
# TK

## Explain the use of the ROC curve and the meaning of the area under the ROC curve.
# TK

## Compare the results obtained with each one of the classifiers, referring to the confusion matrix and associated metrics. Are the results similar? If not, how statistically different are they? If different, what is the reason? If similar, how would you decide to use one method or another?
# TK

# References
# https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
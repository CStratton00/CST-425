# SVM Classifier

# Load the appropriate software packages.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score


# Import the data
colnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '50k-Prediction']
df = pd.read_csv("adult.data", names=colnames)

## Explore the data
print(df.head(20))

# Data preprocessing

## Step 1: Remove unnecessary variables
# TK

## Step 2: Remove duplicate data rows
df.drop_duplicates()

## Step 3: Convert categorical variables into numerical data using a simple label encoder
le = LabelEncoder()

### Use the .fit_transform() function to turn columns with categorical values into columns with numerical value
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

## Subset the data into x and y and set the sample amount
n = 1000 # Number of data points to include in the set (x and y vectors)
x = df.iloc[:n,:-1] # all columns except 50k Prediction column (or classifier)
y = df.iloc[:n,-1] # only the 50k Prediction column (or classifier)

# Data Normalization
x = preprocessing.scale(x)
x = pd.DataFrame(x)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=1234)  # 70% training and 30% test

# Determine the optimal gamma value using the elbow method to create an error graph
# The biggest bend in the elbow determines which number of neighbors creates greatest difference in error reduction
gammaError = []  # will keep track of Error percentage for each n value
gammaValues = [0.01, 1, 5]
for g in range(3):  # calculate 3 different error values
    svcData = SVC(kernel='rbf', gamma=gammaValues[g])  # Uses SVC to model the data with gamma
    svcData = svcData.fit(X_train,y_train)  # Use training data to create a model
    y_pred = svcData.predict(X_test)  # Predict the y values with x_test values
    gammaError.append(1-accuracy_score(y_test,y_pred))  # Compare the y_pred values to the y_test actual values to find Error

plt.plot(range(3),gammaError) #Plot the 3 different error calculations
plt.title("Testing Gamma values 0.01, 1, and 5")
plt.xlabel("Gamma")
plt.ylabel("Error")
plt.show()

# will print the index value where the error is the lowest. Each time the data is randomly selected, so it will change each time it is run
best_g = gammaValues[gammaError.index(min(gammaError))]
print(f"Lowest Error for gamma is: {best_g}")

# Determine the optimal cpenalty value using the elbow method to create an error graph
# The biggest bend in the elbow determines which number of neighbors creates greatest difference in error reduction
cpenaltyError = []  # will keep track of Error percentage for each n value
cpenaltyValues = [1, 10, 100, 1000, 10000]
for c in range(5):  # calculate 5 different error values
    cpenalty = cpenaltyValues[c]
    svcData = SVC(kernel='rbf', C=cpenalty)  # Uses SVC to model the data with cpenalty
    svcData = svcData.fit(X_train,y_train)  # Use training data to create a model
    y_pred = svcData.predict(X_test)  # Predict the y values with x_test values
    cpenaltyError.append(1-accuracy_score(y_test,y_pred))  # Compare the y_pred values to the y_test actual values to find Error

plt.plot(range(5),cpenaltyError) #Plot the 3 different error calculations
plt.title("Using C-Penalty Values 1, 10, 100, 1000, and 10000")
plt.xlabel("C-Penalty")
plt.ylabel("Error")
plt.show()

# will print the index value where the error is the lowest. Each time the data is randomly selected, so it will change each time it is run
best_c = cpenaltyValues[cpenaltyError.index(min(cpenaltyError))]
print(f"Lowest Error is with n neighbor value: {best_c}")

## Build KFold Model to split and build model using the best values
print("\n--- RBF KERNEL ---\n")
k = 5 # Do 5 folds, and split the set into 80/20
kfold = KFold(n_splits=k, random_state=1234, shuffle=True) # Use KFolding using seed, shuffle, and k splits

# Create an RBF SVC model with the best gamma and cpenalty value
svclassifier = SVC(kernel='rbf', gamma=best_g, C = best_c) # Note the use of 'rbf'

acc_score = []

# Use a loop to start the k folding method
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

## Run the final model (make predictions).
y_pred = svclassifier.predict(x_test)

## Display classification results (quantitative and visual)

## Provide the confusion matrix for each classifier.
print("\nCONFUSION MATRIX:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

from mlxtend.plotting import plot_decision_regions
x_test = np.array(x_test)
y_test = np.array(y_test)

value = 1
width = 1.5

for i in range(0,14):
    figure = plot_decision_regions(x_test, y_test, clf=svclassifier, legend=2, feature_index=[i, i+1], filler_feature_values={
}, filler_feature_ranges=width)
    figure.show()

## For each classifier, compute the accuracy, sensitivity, and specificity.
# TK

## Explain the use of the ROC curve and the meaning of the area under the ROC curve.
# TK

# References
# https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
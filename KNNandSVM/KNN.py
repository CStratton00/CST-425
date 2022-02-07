import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,plot_confusion_matrix


# Import the data

# The column names are not included in the .data file so we must create an array of the column names and add them
colnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '50k-Prediction']

# Create a panda dataframe from adult.data and add column names
df = pd.read_csv("adult.data", names=colnames)

# Print out the top 20 instances from the dataframe
print(df.head(20))


# Data preprocessing

# Remove duplicate data rows
df.drop_duplicates()

# Label Encoder is used to take all of our string data and asssign it a unique integer value
le = LabelEncoder()

# Using .fit_transform function to fit label
# encoder and return encoded label
# example, male/female will be turned into 0 and 1 for knn to calculate distance from
df['workclass'] = le.fit_transform(df['workclass'])
df['education'] = le.fit_transform(df['education'])
df['marital-status'] = le.fit_transform(df['marital-status'])
df['occupation'] = le.fit_transform(df['occupation'])
df['relationship'] = le.fit_transform(df['relationship'])
df['race'] = le.fit_transform(df['race'])
df['sex'] = le.fit_transform(df['sex'])
df['native-country'] = le.fit_transform(df['native-country'])
df['50k-Prediction'] = le.fit_transform(df['50k-Prediction'])

# Print out the top 20 instances from the dataframe to show how LabelEncoder changed our data
print(df.head(20))

# Uncomment the .iloc[] lines if the program is running too slow. Will reduce dataset by 2/3.
x = df.drop('50k-Prediction', axis=1)   # all columns except 50k Prediction column (or classifier)
x10000 = x.iloc[:10000,].values              # Set sample amount to first 10000 rows

y = df['50k-Prediction']                # only the 50k Prediction column (or classifier),
y10000 = y.iloc[:10000,].values              # Set sample amount to first 10000 rows

# Need to normalize our data so 0/1 values are weighted the same as 0-50 values
# Dont need to normalize our y data because it is only 0s and 1s.
x = preprocessing.scale(x)
x10000 = preprocessing.scale(x10000)

# Because KNN is a supervised classifier, split dataset into training set and test set
# The X_train and y_train variables will be the 70% of the data, while X_test and y_test will be the 30%.
# X_train and y_train will create the model, then X_test will be plugged into the model which will produce the prediction for y_test values.
# Then we will compare the y_prediction values to the y_test values to determine the accuracy.
# We use x10000 and y10000 becuase these variables will be used 30 times to calculate the optimal neighbor value, hence we need a smaller training set so it can runn efficiently
X_train, X_test, y_train, y_test = train_test_split(x10000, y10000, test_size=0.3,random_state=1234)  # 70% training and 30% test


# Determine the optimal amount of clusters using Error graph
# The biggest bend in the elbow determines which number of neighbors creates greatest difference in error reduction
Error = []  # will keep track of Error percentage for each n value
for n in range(1,31):  # calculate 40 different error values
    knnData = KNeighborsClassifier(n_neighbors=n)  # Uses KNeighborsClassifier to
    knnData = knnData.fit(X_train,y_train)  # Use train data to create a model
    y_pred = knnData.predict(X_test)  # Predict the y values with x_test values
    Error.append(1-accuracy_score(y_test,y_pred))  # Compare the y_pred values to the y_test actual values to find Error

plt.plot(range(1,31),Error) #Plot the 30 different error calculations
plt.title("Using KNeighborsClassifier with neighbor values 1-31")
plt.xlabel("Number of neighbors")
plt.ylabel("Error")
plt.show()

# will print the index/n_neighbor value where the error is the lowest. Each time the data is randomly selected, so it will change each time it is run
best_n = Error.index(min(Error))
print(f"Lowest Error is with n neighbor value: {best_n}")


# Main KNN Calculation:

#Show that x and y are full data set -> 32k instances
print(f"Length of x data = {len(x)}")
print(f"Length of y data = {len(y)}")

# Recreate X_train, X_test, y_train, y_test values with the full data set for the one knn calculation
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)  # 70% training and 30% test

# Creates a single KNN model with optimal n value and all 32k data points
knn = KNeighborsClassifier(n_neighbors=best_n)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

# Create a confusion matrix of the y_test values(correct answers) to the y_pred values(predicted values)
confusion = confusion_matrix(y_test, y_pred)
print(confusion)
# Results from confusion matrix:
print("6922 true positives")
print("1158 true negatives")
print("547 false negatives")
print("1142 false positives")



# Precision-
# How correct the prediction was

# Recall-
# A measure of the models completeness

# f1-score-


# Support-


print(classification_report(y_test,y_pred))



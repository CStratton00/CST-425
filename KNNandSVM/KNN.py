import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,plot_confusion_matrix

#read data as csv and print out the top 20 results
# Import the data
colnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '50k-Prediction']
df = pd.read_csv("adult.data", names=colnames)
print(df.head(20))


# Data preprocessing
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



x = df.drop('50k-Prediction', axis=1)  # all columns except 50k Prediction column (or classifier)
x = x.iloc[:,].values  # Set sample amount to first 1000 rows
y = df['50k-Prediction'].iloc[:,].values  #only the 50k Prediction column (or classifier), Set sample amount to first 1000 rows

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)  # 70% training and 30% test




#Determine the optimal amount of clusters using Error graph
#The biggest bend in the elbow determines which number of clusters creates greatest difference in error reduction
#From the graph, k=3 is shown to be the optimal number of cluster
Error = []
for i in range(1,41):
    knnData = KNeighborsClassifier(n_neighbors=i)
    knnData = knnData.fit(X_train,y_train)
    y_pred = knnData.predict(X_test)
    Error.append(1-accuracy_score(y_test,y_pred))

plt.plot(range(1,41),Error)
plt.title("Using KNeighborsClassifier with neighbor values 1-41")
plt.xlabel("Number of neighbors")
plt.ylabel("Error")
plt.show()

#will print the index/n_neighbor value where the error is the lowest. Each time the data is randomly selected, so it will change each time it is run
best_n = Error.index(min(Error))
print(best_n)

#
knn = KNeighborsClassifier(n_neighbors=best_n)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
confusion = confusion_matrix(y_test, y_pred)
print(confusion)

print(classification_report(y_test,y_pred))



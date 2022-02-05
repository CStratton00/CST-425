import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

#read data as csv and print out the top 20 results
# Import the data
colnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '50k-Prediction']
df = pd.read_csv("adult.data", names=colnames)
print(df.head(20))


# Data preprocessing

le =


print(df.head(20))

x = df.iloc[1:,[0,2,4,10,11,12]].values

Error = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)

plt.plot(range(1,11),Error)
plt.show()


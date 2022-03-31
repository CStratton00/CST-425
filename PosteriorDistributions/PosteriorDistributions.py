import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import networkx as nx

# Import data
df = pd.read_csv("./syntheticdata.csv", header=None)
print(df)

# row and column headers list
headers = ["Museum", "Concert", "Sports Event", "Restaurant", "Hike"]

# Convert into a probabilistic matrix, and then convert the probabilistic matrix into a transition matrix
# NOTE: See if we could do this automatically, without having to manually input data. Let's just use this to verify our result
transitionMatrix = df.div(df.sum(axis=1), axis=0)

transitionMatrixNP = transitionMatrix.to_numpy()

# function to iterate over a matrix and plot the values
def markovPlot(matrix, x, y, n):
    xval = np.linspace(0, 100, 100)
    yval = []
    for k in range(1, n):
        yval.append(pow(matrix, k)[x, y] * 100)
    plt.xlabel("Days")
    plt.ylabel("% Chance of happening")
    plt.plot(xval, yval)
    plt.title("{} to {}".format(headers[x], headers[y]))
    plt.show()


# for i, j in transitionMatrix.iterrows():
#     markovPlot(transitionMatrixNP, i, j, 100)

for i in range(transitionMatrixNP.shape[0]):
    for j in range(transitionMatrixNP.shape[1]):
        markovPlot(transitionMatrixNP, i, j, 101)


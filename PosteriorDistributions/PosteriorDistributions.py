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
print(transitionMatrixNP)
# function to iterate over a matrix and plot the values
def markovPlot(matrix, x, y, n):
    xval = np.linspace(0, 10, 10)
    yval = []
    day = 0
    ssv = 0
    for k in range(1, n):
        yval.append(pow(matrix, k)[x, y] * 100)

    for i in range(len(yval)-1):
        change = yval[i] - yval[i + 1]
        if change < .05:
            day = i
            ssv = round(yval[i], 3)
            break

    plt.xlabel("Days")
    plt.ylabel("% Chance of happening")
    plt.plot(xval, yval)
    plt.title("{} to {} \n Steady at day {} with a value of {}".format(headers[x], headers[y], day, ssv))
    plt.show()


# for i, j in transitionMatrix.iterrows():
#     markovPlot(transitionMatrixNP, i, j, 100)

# Generate graphs from transition matrix to illustrate the Markov Chain
# for i in range(transitionMatrixNP.shape[0]):
#     for j in range(transitionMatrixNP.shape[1]):
#         markovPlot(transitionMatrixNP, i, j, 11)


concertHikeRestaurant = transitionMatrixNP[1][4] * transitionMatrixNP[4][3]
print("Percent odds of Concert to Hike to Restaurant = ", concertHikeRestaurant * 100)

trans5 = pow(transitionMatrixNP, 5)
sum_arr = trans5.sum(axis=0)

museum5 = sum_arr[0]
concert5 = sum_arr[1]
sportsEvent5 = sum_arr[2]
restaurant5 = sum_arr[3]
hike5 = sum_arr[4]

print("\nPercent odds of visiting the Museum 5th = ", museum5 * 100)
print("Percent odds of visiting the Concert 5th = ", concert5 * 100)
print("Percent odds of visiting the Sports Event 5th = ", sportsEvent5 * 100)
print("Percent odds of visiting the Restaurant 5th = ", restaurant5 * 100)
print("Percent odds of visiting the Hike 5th = ", hike5 * 100)

import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import networkx as nx
import pymc3 as pm

# Import data
df = pd.read_csv("./syntheticdata.csv", header=None)
# print(df)

# row and column headers list
headers = ["Museum", "Concert", "Sports Event", "Restaurant", "Hike"]

# Convert into a probabilistic matrix, and then convert the probabilistic matrix into a transition matrix
transitionMatrix = df.div(df.sum(axis=1), axis=0)

transitionMatrixNP = transitionMatrix.to_numpy()
print(transitionMatrixNP)
# function to iterate over a matrix and plot the values
def markovPlot(matrix, x, y, n):
    xval = np.linspace(0, 10, 10)
    yval = []
    day = 0
    ssv = 0
    mt = matrix
    for k in range(1, n):
        mt = np.dot(mt, matrix)
        yval.append(mt[x, y] * 100)


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


for i, j in transitionMatrix.iterrows():
    markovPlot(transitionMatrixNP, i, j, 100)

# Generate graphs from transition matrix to illustrate the Markov Chain
for i in range(transitionMatrixNP.shape[0]):
    for j in range(transitionMatrixNP.shape[1]):
        markovPlot(transitionMatrixNP, i, j, 11)

# Step 7:
concertHikeRestaurant = transitionMatrixNP[1][4] * transitionMatrixNP[4][3]
# print("Percent odds of Concert to Hike to Restaurant = ", concertHikeRestaurant * 100)

# Step 8:
# find the probabilities of the transition matrix on the 5th step
transDot = transitionMatrix
for i in range(4):  # Four transitions, so run 4 times
    transDot = np.dot(transDot, transitionMatrix)

# starting probability vector
spv = np.array([.2, .2, .2, .2, .2])

# sum the columns of the fifth step
sum_trans = transDot.sum(axis=0)

# multiply the starting probability vector by the sum of the columns
spv_tran = sum_trans * spv

# print the likelihood of visiting any of the locations as the fifth step
# print("\nPercent odds of visiting the Museum 5th = ", spv_tran[0] * 100)
# print("Percent odds of visiting the Concert 5th = ", spv_tran[1] * 100)
# print("Percent odds of visiting the Sports Event 5th = ", spv_tran[2] * 100)
# print("Percent odds of visiting the Restaurant 5th = ", spv_tran[3] * 100)
# print("Percent odds of visiting the Hike 5th = ", spv_tran[4] * 100)
#
# plt.pie(spv_tran, labels=headers, autopct='%1.1f%%')
# plt.show()

# Step 9: Format a question that requires a MCMC to answer
# Question: What is the most visited location in town after 5 days?

# Step 10:

# METROPOLIS-HASTINGS
# Define the model
# with pm.Model() as model:
#     # Define the prior
#     # Prior is a matrix of probabilities
#     x = pm.Normal('x', mu=0, sigma=1)
#
# with model:
#     # Define beta and alpha
#     b = pm.Beta('b', alpha=1, beta=1, shape=(5, 5))
#
# xmin = -3
# xmax = 3
# pmax = 0.8
# N_MC = 100000
#
# t = np.random.normal(xmin, xmax, N_MC)  # get uniform temporary x values
# y = np.random.normal(0, pmax, N_MC)  # get uniform random y values
# # plot all the t-y pairs
# plt.scatter(t, y, s=0.1, c='orange')
# #make a mask that keeps index to the accepted pairs. Plot them
# mask = y < model
# plt.scatter(t[mask], y[mask], s=0.1)
# accept_prob = t[mask].size / t.size
# #histogram the t values with and without the mask
# _ = plt.hist(t, bins=100)
# _ = plt.hist(t[mask], bins=100)

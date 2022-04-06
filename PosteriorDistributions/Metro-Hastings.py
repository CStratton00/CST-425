import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import networkx as nx
import pymc3 as pm
from IPython.core.pylabtools import figsize

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
    newmatrix = np.dot(matrix, matrix)
    yval.append(newmatrix[x, y] * 100)

    for k in range(1, n-1):
        newmatrix = np.dot(newmatrix, matrix)
        yval.append(newmatrix[x, y] * 100)

    for i in range(len(yval)-1):
        change = abs(yval[i] - yval[i + 1])
        if change < .01:
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
# Question: What is the distribution of visited locations in town after 5 days?
# And what is the most visited location after 5 days?

# Step 10: METROPOLIS-HASTINGS
# 1. Start with a random sample
# 2. Determine the probability density associated with the sample
# 3. Propose a new, arbitrary sample (and determine its probability density)
# 4. Compare densities (via division), quantifying the desire to move
# 5. Generate a random number, compare with desire to move, and decide: move or stay
# 6. Repeat until the number of iterations is reached and the convergence criteria is met

# Create a random training dataset
def randomDataset(n):
    choicesList = [1.0, 2.0, 3.0, 4.0, 5.0]
    return np.random.choice(choicesList, size=n, replace=True, p=spv_tran)


training = np.array(randomDataset(1000))  # Create a random dataset with 1000 observations

# Create the model with initial conditions
with pm.Model() as model:
    # Define the prior parameters
    theta = pm.Beta("Theta", alpha=0, beta=1)  # Define a basic beta distribution for the prior distribution
    observed = pm.Normal("Observed", mu=0, sigma=1, observed=training)  # Define likelihood using a normal Gaussian distribution with observed data

# Perform Metropolis-Hastings sampling
num_samples = 10000  # Set the number of total samples to take

with model:
    start = [3.0]  # Set the initial condition
    step = pm.Metropolis()  # Use sampling with Metropolis-Hastings steps
    trace = pm.sample(num_samples, step=step, start=start, random_seed=456, return_inferencedata=False)  # Perform sampling

# Plot the Metropolis-Hastings results
p_true = 0.5
figsize(12.5, 4)
plt.title(r"Posterior distribution of $\theta for sample size N=1000$")
plt.vlines(p_true, 0, 25, linestyle='--', label="true $\theta$ (unknown)", color='red')
plt.hist(trace["Theta"], bins=25, histtype='stepfilled', density=True, color='#348ABD')
plt.legend()
plt.show()

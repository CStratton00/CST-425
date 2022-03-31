import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import networkx as nx
import pymc3 as pm

# METROPOLIS-HASTINGS
# Define the model
with pm.Model() as model:
    # Define the prior
    # Prior is a matrix of probabilities
    x = pm.Normal('x', mu=0, sigma=1)

with pm.Model():
    # Define beta and alpha
    b = pm.Beta('b', alpha=1, beta=1, shape=(5, 5))

print(x)
print(b)

# Function to perform Gaussian sampling
def gaussianSample(mu, sigma):
    return np.random.normal(mu, sigma)

def accept_reject(N):
    min = -1
    max = 1


# Function to perform Metropolis-Hastings sampling
def metropolisHastings(mu, sigma, n):
    # Initialize the chain
    chain = [mu]
    # Initialize the proposal distribution
    proposalDistribution = gaussianSample(mu, sigma)
    # Initialize the acceptance rate
    acceptanceRate = 0
    # Iterate over the chain
    for i in range(1, n):
        # Sample from the proposal distribution
        proposalDistribution = gaussianSample(mu, sigma)
        # Calculate the acceptance rate
        acceptanceRate = np.exp(proposalDistribution - mu) / (
                    np.exp(proposalDistribution - mu) + np.exp(mu - proposalDistribution))
        # Accept the proposal
        if acceptanceRate > np.random.uniform():
            mu = proposalDistribution
        # Add the sample to the chain
        chain.append(mu)
    return chain


x = np.linspace(0, 100, 100)
metro = metropolisHastings(0, 1, 100)


print(x)
plt.plot(x, metro)
plt.show()
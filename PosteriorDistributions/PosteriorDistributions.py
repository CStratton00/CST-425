import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import networkx as nx

# Import data
df = pd.read_csv("./syntheticdata.csv", header=[0], )
print(df)

# Convert into a probabilistic matrix, and then convert the probabilistic matrix into a transition matrix
# NOTE: See if we could do this automatically, without having to manually input data. Let's just use this to verify our result
transitionMatrix = np.matrix(
    [[1/15, 1/5, 1/3, 1/5, 1/5],
    [7/21, 1/21, 4/21, 6/21, 3/21],
    [8/21, 2/21, 2/21, 5/21, 4/21],
    [5/18, 4/18, 5/18, 3/18, 1/18],
    [2/13, 4/13, 1/13, 5/13, 1/13]]
                             )

print(transitionMatrix)



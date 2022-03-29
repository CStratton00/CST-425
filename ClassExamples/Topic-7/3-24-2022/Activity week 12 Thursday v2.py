# Come up with a scenario, e.g. weather, with 4 states: sunny, rainy, windy, cloudy.

# Define a transition matrix and plot the DFA

# Calculate probabilities like:
# - P(day 15 being sunny)
# - P (3 cloudy days, starting with day 20)

# Suggested tools: Python with numpy for matrix calculations

# Pycharm to Pycharm
# Pycharm to RStudio
# Rstudio to Pycharm
# Rstudio to Rstudio

#          | Pycharm | Rstudio | VSCode | Notepad++
# Pycharm  |   .6    |    .1   |   .2   |   .1    |
# Rstudio  |   .1    |    .4   |   .2   |   .3    |
# VSCode   |   .4    |    .1   |   .4   |   .1    |
# Notepad++|   .1    |    .3   |   .3   |   .3    |

import numpy as np
import matplotlib.pyplot as plt

labels = ["Pycharm", "Rstudio", "VSCode", "Notepad++"]
a = np.matrix('.6 .1 .2 .1; .1 .4 .2 .3; .4 .1 .4 .1; .1 .3 .3 .3')
print(a)


# function to iterate over a matrix and plot the values
def markovPlot(x, y, n):
    xval = np.linspace(0, 100, 100)
    yval = []
    for k in range(1, n):
        yval.append(pow(a, k)[x, y])

    plt.plot(xval, yval)
    plt.title("{} to {}".format(labels[x], labels[y]))
    plt.show()


for i in range(0, 4):
    for j in range(0, 4):
        markovPlot(i, j, 101)

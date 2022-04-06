import numpy as np
import matplotlib.pyplot as plt


# function to iterate over a matrix and plot the values
def markovPlot(matrix, x, y, n):
    # row and column headers list
    headers = ["Museum", "Concert", "Sports Event", "Restaurant", "Hike"]
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
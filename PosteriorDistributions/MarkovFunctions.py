import numpy as np
import matplotlib.pyplot as plt


# function to iterate over a matrix and plot the values
def markovPlot(matrix, headers, x, y, n):
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

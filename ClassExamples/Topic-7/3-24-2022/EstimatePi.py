import random
from numba import jit,cuda
#import numpy as np
#inCircle = np.empty()
#outCircle = np.empty()


@jit(target_backend="cuda")
def func2():
    inpoint = 0
    outpoint = 0
    for i in range(1000000):
        x = random.random()
        y = random.random()

        if pow(x, 2) + pow(y, 2) < 1:
            inpoint += 1
        else:
            outpoint += 1
    return inpoint, outpoint


if __name__ == "__main__":

    inpointR, outpointR = func2()
    ratio = inpointR / (inpointR + outpointR)
    print(f"{inpointR} / {inpointR+outpointR} = ", ratio)
    print(ratio/(pow(.5, 2)))


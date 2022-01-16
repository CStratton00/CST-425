import numpy as np

np.random.seed(100)

n = 1
p = .5
size = 100

x = np.random.binomial(n, p, size)
probs_100 = [np.equal(x, i).mean() for i in range(n)]
print(probs_100)

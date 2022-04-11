import pymc3 as pm
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize

N=1000 #the number of samples
occurences=np.random.binomial(1, p=0.5, size=N)
k=occurences.sum() #the number of head
#fit the observed data
with pm.Model() as model1:
    theta=pm.Uniform('theta', lower=0, upper=1)
with model1:
    obs=pm.Bernoulli("obs", theta, observed=k)
    step=pm.Metropolis()
    trace=pm.sample(18000, step=step)
    burned_trace1=trace[1000:]

#plot the posterior distribution of theta.
p_true=0.5
figsize(12.5, 4)
plt.title(r"Posterior distribution of $\theta for sample size N=1000$")
plt.vlines(p_true,0, 25, linestyle='--', label="true $\theta$ (unknown)", color='red')
plt.hist(burned_trace1["theta"], bins=25, histtype='stepfilled', density=True, color='#348ABD')
plt.legend()
plt.show()
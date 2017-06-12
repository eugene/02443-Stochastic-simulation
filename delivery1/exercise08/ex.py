from bootstrap import bootstrap, estimate_p, get_median, get_mean
import numpy as np
import math
import random
from scipy.stats import pareto
import matplotlib.pyplot as plt

'''
Ex 13 / chapter 7

a) Explain how we can use the bootstrap approach to estimate p
Since we don't know the true mean of the distribution, the best we can do is estimate it just by taking the mean of the sample we have and use that values for mu.
Next, assuming that the sample we have represents the whole population, we can bootstrap it (draw numbers with replacement) and thus generate new samples.
These can be then used for calculating the sum(X/N) which is the mean of the bootstrapped sample.
Afterwards, we count how many times the value of sum(X/N) - mu lies between 'a' and 'b' and divide by the number of bootstraps we generated to get the probability.

b) p = 0.76
'''
x = [56,101,78,67,93,87,64,72,80,69]
reps = 100
b = bootstrap(x,reps)
p = estimate_p(x, b, -5, 5)

'''
Calculate the mean and variance with their variances for Pareto distribution
'''
beta = 1
k = 1.05
N = 2000

x = [beta*(math.pow(random.random(), (-1/k))) for i in range(N)]
reps = 1000
b = bootstrap(x,reps)

median, median_var = get_median(x,b)
mean, mean_var = get_mean(x,b)
print 'Mean:',mean,' mean variance',mean_var,"\nMedian:",median,' median variance',median_var

'''
Mean: 7.30978951372  mean variance 7.41035913387
Median: 1.83785492957  median variance 0.02601195163

The mean has much higher variance, which is cause by the fact that it is much more affected by the outliers / less likely values on the right tail of the distribution.
The median is being shifted by 1 sample whereas the mean is affected by the actual value of the sample.
This shows that bootstrapping can't be blindly used for reduction of all kind of variances and its performance is problem dependent.
'''

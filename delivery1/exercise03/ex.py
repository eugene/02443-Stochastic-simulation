from __future__ import division
from continous_distributions import exponential, box_muller, pareto, normal, plot_pdf, plot_CIs_series, plot_CIs_lines, chi2_test
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import exponential as exponential_np, normal as normal_np, pareto as pareto_np
from scipy.stats import norm, pareto as pareto_sc, expon

# Setup
n = 10000
alpha = 0.05
bins = 100

# Exponential distribution
rate = 9
x = exponential(rate, n)
hist,edges = np.histogram(x,bins)
expected = [int(n*(expon.cdf(edges[i+1],scale=1/rate) - expon.cdf(edges[i],scale=1/rate))) for i in range(len(edges)-1)]
chi2_test(x,expected,alpha,bins)

# Box muller
x,y = box_muller(n,True)

# Normal
x,_,_ = normal(n, alpha=0.05, reps=1, plot=True)
hist,edges = np.histogram(x,bins)
expected = [int(n*(norm.cdf(edges[i+1],0,1) - norm.cdf(edges[i],0,1))) for i in range(len(edges)-1)]
chi2_test(x,expected,alpha,bins)

# Pareto distribution
beta = 1; k=2.05
print 'Pareto\n','beta=',beta,'k=',k
x=pareto(beta, k=k, n=n, plot=True)
hist,edges = np.histogram(x,bins)
expected = [int(n*(pareto_sc.cdf(edges[i+1],k) - pareto_sc.cdf(edges[i],k))) for i in range(len(edges)-1)]
chi2_test(x,expected,alpha,bins)

k=2.5
print 'Pareto\n','beta=',beta,'k=',k
pareto(beta, k=k, n=n, plot=True)
hist,edges = np.histogram(x,bins)
expected = [int(n*(pareto_sc.cdf(edges[i+1],k) - pareto_sc.cdf(edges[i],k))) for i in range(len(edges)-1)]
chi2_test(x,expected,alpha,bins)

k=3
print 'Pareto\n','beta=',beta,'k=',k
pareto(beta, k=k, n=n, plot=True)
hist,edges = np.histogram(x,bins)
expected = [int(n*(pareto_sc.cdf(edges[i+1],k) - pareto_sc.cdf(edges[i],k))) for i in range(len(edges)-1)]
chi2_test(x,expected,alpha,bins)

k=4
print 'Pareto\n','beta=',beta,'k=',k
pareto(beta, k=k, n=n, plot=True)
hist,edges = np.histogram(x,bins)
expected = [int(n*(pareto_sc.cdf(edges[i+1],k) - pareto_sc.cdf(edges[i],k))) for i in range(len(edges)-1)]
chi2_test(x,expected,alpha,bins)

# 95% confidence intervals for mean and variance
x, mean_ci, var_ci = normal(10, alpha=0.05, reps=100, plot=False)
plot_CIs_series(mean_ci,0,'Mean')
plot_CIs_series(var_ci,1,'Variance')
plt.show()



# Chi2 for exponential :
# Critical value: 16.9189776046  chi2-stat: 5.11718121725
# The goodness of fit test has been therefore passed
#
# Chi2 normal :
# Critical value: 16.9189776046  chi2-stat: 8.68033473907
# The goodness of fit test has been therefore passed
#
# Pareto
# beta= 1 k= 2.05
# Mean:  1.98359728891  true mean: 1.95238095238
# Variance:  10.5466676935  true variance: 37.1882086168
# Critical value: 16.9189776046  chi2-stat: 4.66666666667
# The goodness of fit test has been therefore passed
#
# Pareto
# beta= 1 k= 2.5
# Mean:  1.65401648359  true mean: 1.66666666667
# Variance:  1.65654370423  true variance: 2.22222222222
# Critical value: 16.9189776046  chi2-stat: 33.2005148005
# The goodness of fit test has NOT been passed for values higher than k=2.05, altough the overall shape of the distributions is very similar to the desired one
#
# Pareto
# beta= 1 k= 3
# Mean:  1.5155929964  true mean: 1.5
# Variance:  1.02151823789  true variance: 0.75
# Critical value: 16.9189776046  chi2-stat: 361.090036014
#
# Pareto
# beta= 1 k= 4
# Mean:  1.34085681951  true mean: 1.33333333333
# Variance:  0.22576177862  true variance: 0.222222222222
# Critical value: 16.9189776046  chi2-stat: 0.102420484097
#
# For 100 95% confidence intervals we expect +-95 of them contain the true value
# This value can of course vary slightly, and if we count the actual number of interval containing the true mean=0
# we get 96 for the mean and the variance, which are close to expected values.
#
#
#
#

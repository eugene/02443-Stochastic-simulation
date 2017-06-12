srand(0) # resets the random number generator

using Distributions
using UnicodePlots
using StatsBase

"""
Given value samples `values` from a continuous distribution, compute the
confidence interval with significance level α.

The distribution is assumed Student's-t: the mean and the variance is estimated.
"""
function confidence_interval(values; α=0.05)
    n = length(values)
    df = n - 1
    μ = mean(values)
    S = var(values)

    lo = quantile(TDist(df), α/2)
    hi = quantile(TDist(df), 1-α/2)

    (μ + lo*(sqrt(S/n)), μ, μ + hi*(sqrt(S/n)))
end

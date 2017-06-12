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
    μ = mean(values)
    S = var(values)

    lo, hi = confidence_interval(n, μ, S, α=α)
    lo, μ, hi
end

function confidence_interval(n, μ, S; α=0.05)
    df = n-1;
    lo = quantile(TDist(df), α/2)
    hi = quantile(TDist(df), 1-α/2)

    (μ + lo*(sqrt(S/n)), μ + hi*(sqrt(S/n)))
end

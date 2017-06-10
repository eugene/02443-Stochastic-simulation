# Exercise 1
#
# • Write a program generating 10.000 (pseudo-) random numbers and present these 
#   numbers in a histogram (e.g. 10 classes).
#
# ⋄ First implement the LCG yourself by experimenting with different values 
#   of “a”, “b” and “c”.
# ⋄ Evaluate the quality of the generators by several statistical tests.
# ⋄ Then apply a system available generator (e.g. drand48() C, and C++) and 
#   perform various statistical tests for this also. As a minimum you should 
#   perform a χ2 test and an UP/DOWN run test. Optional supplementary tests 
#   are histograms and Ui, Ui+1 plots, test for zero correlation 
#   and up/down runtest.

include("common.jl")

# Resets Julia MersenneTwister random number generator
srand(0) 

using Distributions
using StatsBase
using UnicodePlots

# Part 1: Write a program generating 10.000 (pseudo-) random numbers and present these 
#         numbers in a histogram (e.g. 10 classes).
#
# julia> histogram(rand(10000), 10)
#              ┌────────────────────────────────────────┐
#    (0.0,0.1] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 1000   │
#    (0.1,0.2] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 1002   │
#    (0.2,0.3] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 973     │
#    (0.3,0.4] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 995    │
#    (0.4,0.5] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 1013   │
#    (0.5,0.6] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 941      │
#    (0.6,0.7] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 1025  │
#    (0.7,0.8] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 972     │
#    (0.8,0.9] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 1066 │
#    (0.9,1.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 1013   │
#              └────────────────────────────────────────┘

# Part2: First implement the LCG yourself by experimenting with different values 
#        of `a`, `b` and `c`.
#

"""
Linear Congruential Generator Implementation
"""
function lcg(x₀, a, c, M; limit = 16, Uᵢ = false)
    r = []
    xₚ = x₀

    while limit > 0
        xₚ = mod(a * xₚ + c, M)
        push!(r, xₚ)
        limit-=1
    end

    if Uᵢ
        r./M
    else
        r
    end
end

# Let's run our LCG on the values from data from slides:
# julia> lcg(3, 5, 1, 16)
# 16-element Array{Any,1}: 0 1 6 15 12 13 2 11 8 9 14 7 4 5 10 3
#
# And also values for a POSIX/glibc jm_rand48 (found on Wikipedia):
# 
# M = 2⁴⁸, a = 0x5deece66d, c = 11
#
# julia> lcg(12345, 0x5deece66d, 11, 2^48, Uᵢ = true)
# julia> lcg(12345, 0x5deece66d, 11, 2^48, Uᵢ = true)
# 16-element Array{Float64,1}: 0.105882, 0.79826, 0.016059, 0.664037
# 0.0429341, 0.99479, 0.845222, 0.217724, 0.276606, 0.418719
# 0.297666, 0.824748, 0.156347, 0.503388, 0.78659, 0.935067
 
"""
T value calculator.  
"""
function T(X::Vector)
    T_acc = 0 # T accomulator 

    bins = counts(X)
    expected = length(X) / length(bins)

    for class = 1:length(bins)
        T_acc += (bins[class] - expected)^2 / expected
    end

    T_acc
end

# Let's generate 10^8 random numbers with the POSIX LCG parameters:
#
# julia> d = lcg(12345, 0x5deece66d, 11, 2^48, limit = 10^8, Uᵢ = true);
#
# Apply normalization so we can fill the bins for the T-test.
# 
# julia> numbers = Array{Int}(floor(d*10^4));
# julia> (length(numbers), minimum(numbers), maximum(numbers))
# (100000000, 0, 9999)
#
# Now, let us see a histogram of those: 
#
# julia> histogram(numbers)
#                  ┌────────────────────────────────────────┐
#      (0.0,500.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 4998730 │
#   (500.0,1000.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 5001299 │
#  (1000.0,1500.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 5002401 │
#  (1500.0,2000.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 4999110 │
#  (2000.0,2500.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 5001803 │
#  (2500.0,3000.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 4999071 │
#  (3000.0,3500.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 4999110 │
#  (3500.0,4000.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 5001835 │
#  (4000.0,4500.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 5001464 │
#  (4500.0,5000.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 5000953 │
#  (5000.0,5500.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 4997302 │
#  (5500.0,6000.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 5001337 │
#  (6000.0,6500.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 4997808 │
#  (6500.0,7000.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 5000483 │
#  (7000.0,7500.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 5001055 │
#  (7500.0,8000.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 4999025 │
#  (8000.0,8500.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 4998357 │
#  (8500.0,9000.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 4998192 │
#  (9000.0,9500.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 5000726 │
# (9500.0,10000.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 4999939 │
#                  └────────────────────────────────────────┘

"""
Χ² Hypothesis evaluator
"""
function hypo_eval(T_val, df)
    high = quantile(Chisq(df), 0.95) 

    if T_val >= high
        return "H₀ REJECTED. T($(T_val)) >= $(high)"
    else
        return "H₀ ACCEPTED."
    end
end

# And evaluate some T-values:
#
# julia> Tv = T(numbers)
# 10080.861400000036
# julia> hypo_eval(Tv, 9999 - 1)
# "H₀ ACCEPTED."

#
# Our next goal is to do a Runs test

"""
Runs test
"""
function runs_test(X)
    r = 0
    runs = []
    for i = 1:length(X) - 1
        if X[i+1] < X[i]
            push!(runs, -1)
        else
            push!(runs, 1)
        end
    end

    for j = 1:length(runs) - 1
        (runs[j] != runs[j+1]) && (r += 1)
    end

    r
end

# julia> length(numbers)
# 100000000
# julia> runs_test(numbers)
# 66660541

"""
Runs test evaluator where `n` is the length of the sequens
and `r` is total number of runs (runs_test function)
"""
function runs_evaluator(n, r)
    Z = (r - ((2*n - 1)/3))/sqrt((16*n - 29)/90)

    low  = quantile(Normal(), 0.025) 
    high = quantile(Normal(), 0.975) 

    if Z <= low
        return "H₀ REJECTED. Z: $(Z)) <= $(low)"
    elseif Z >= high
        return "H₀ REJECTED. Z: $(Z)) >= $(high)"
    else
       return "H₀ ACCEPTED." 
    end
end

#
# Finally, we run the test for Julia build in random generator, a Me-
# rsenneTwister.
#

# Excercise 2
# Discrete random variables
#
# In the excercise you can use a build in procedure for 
# generating uniform random variables. Compare the results obtained
# in simulations with expected results. Use histograms (and tests).
#
# • Choose a value for p in the geometric distribution and simulate 
#   10,000 outcomes.
# 
# • Simulate the 6 point distribution with
#   
#    X  | 1    | 2    | 3    | 4    | 5    | 6    |
#   ------------------------------------------------  
#    Pᵢ | 7/48 | 5/48 | 1/8  | 1/16 | 1/4  | 5/16 |
#
#   – by applying a direct (crude) method
#   – by using the Alias mehtod
#   – by using the the rejection method

include("common.jl")

##################################
# PART 1: Geometric distribution #
##################################

# The probability of first success occuring on 
# the n'tn trial given probability `p`
Geometric(n; p = 0.5) = p * (1-p)^(n-1)

# The CDF of that Geometric distribution is:
Geometric_CDF(n; p = 0.5) = 1 - (1 - p)^n
# which is the probability that `X` will take the 
# value less than or equal to `n`. 

# The inverse of that CDF is:
Geometric_CDF_inv(U; p = 0.5) = floor(log(U)/log(1-p)) + 1
# which we use to to get a sample driven by `U` from the Geometric
# distribution given a `p`.


# julia> simulation = [Geometric_CDF_inv(rand()) for i = 1:10_000]
# julia> histogram(simulation)
#             ┌────────────────────────────────────────┐
#   (1.0,2.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 4993 │
#   (2.0,3.0] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 2509                  │
#   (3.0,4.0] │▇▇▇▇▇▇▇▇▇ 1261                          │
#   (4.0,5.0] │▇▇▇▇ 626                                │
#   (5.0,6.0] │▇▇ 317                                  │
#   (6.0,7.0] │▇ 145                                   │
#   (7.0,8.0] │ 64                                     │
#   (8.0,9.0] │ 41                                     │
#  (9.0,10.0] │ 22                                     │
# (10.0,11.0] │ 13                                     │
# (11.0,12.0] │ 5                                      │
# (12.0,13.0] │ 0                                      │
# (13.0,14.0] │ 1                                      │
# (14.0,15.0] │ 1                                      │
# (15.0,16.0] │ 1                                      │
# (16.0,17.0] │ 1                                      │
#             └────────────────────────────────────────┘


#############################################
# PART 2: Simulate the 6 point distribution #
#############################################

pᵢₛ = [7//48, 5//48, 1//8, 1//16, 1//4, 5//16]

"""
Crude method. 

Note: Performance is improved as we sort the pᵢₛ in 
      descending order.
"""
function crude()
    U = rand()
    acc = 0
    sorted = sort(pᵢₛ)

    for i = 1:length(sorted)
        acc += sorted[i]
        U <= acc && return sorted[i]
    end
end

# julia> simulation = [crude() for i = 1:10_000]
# julia> histogram(simulation)
#             ┌────────────────────────────────────────┐
# (0.06,0.08] │▇▇▇▇▇▇▇ 635                             │ (pᵢ = 1/16)
#  (0.1,0.12] │▇▇▇▇▇▇▇▇▇▇▇ 1029                        │ (pᵢ = 5/48)
# (0.12,0.14] │▇▇▇▇▇▇▇▇▇▇▇▇▇ 1203                      │ (pᵢ = 1/8)
# (0.14,0.16] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 1467                   │ (pᵢ = 7/48)
# (0.24,0.26] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 2546       │ (pᵢ = 1/4)
#  (0.3,0.32] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 3120 │ (pᵢ = 5/16)
#             └────────────────────────────────────────┘
#
# Benchmark
# julia> @time [crude() for i = 1:10_000]
#   0.059389 seconds (359.96 k allocations: 12.352 MB)


"""
Alias method, takes parameter n (how many samples you need, default 16).
Implemented with the help of description found here: 
    https://pandasthumb.org/archives/2012/08/lab-notes-the-a.html
"""
type AliasLookup
    # Lookup is a data structure consisting of
    #  • pᵢₛ index into first element
    #  • pᵢₛ index into second element
    #  • cutoff (split) point between two

    target::Int64
    alias::Int64
    cutoff::Rational{Int64}
end

function alias(n = 16)
    line = 1 // length(pᵢₛ)
    
    lookup = map(1:length(pᵢₛ)) do i
        AliasLookup(i, -1, pᵢₛ[i])
    end

    small = find(e -> e < line, pᵢₛ)
    large = find(e -> e > line, pᵢₛ)

    while length(large) > 0
        largeₑ = pop!(large)
        smallₑ = pop!(small)

        source = lookup[largeₑ]
        target = lookup[smallₑ]
        target.alias = largeₑ
        available_space = line - target.cutoff
        source.cutoff -= available_space

        if source.cutoff > line
            push!(large, largeₑ)
        elseif source.cutoff < line
            push!(small, largeₑ)
        else 
            # source is balanced
        end
    end

    ret = Vector{Rational}(n)
    for i = 1:n
        u₁ = convert(Int, ceil(rand()*length(pᵢₛ)))
        u₂ = line*rand()
        
        index = lookup[u₁]

        if u₂ <= index.cutoff
            ret[i] = pᵢₛ[index.target]
        else
            ret[i] = pᵢₛ[index.alias]
        end
    end

    ret
end
# julia> simulation = alias(10_000)
# julia> histogram(simulation)
#             ┌────────────────────────────────────────┐
# (0.06,0.08] │▇▇▇▇▇▇▇ 629                             │ (pᵢ = 1/16)
#  (0.1,0.12] │▇▇▇▇▇▇▇▇▇▇▇ 1022                        │ (pᵢ = 5/48)
# (0.12,0.14] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 1330                    │ (pᵢ = 1/8)
# (0.14,0.16] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 1428                   │ (pᵢ = 7/48)
# (0.24,0.26] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 2490        │ (pᵢ = 1/4)
#  (0.3,0.32] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 3101 │ (pᵢ = 5/16)
#             └────────────────────────────────────────┘
# 
# Benchmark:
# julia> @time alias(10_000)
#   0.006826 seconds (109.04 k allocations: 2.352 MB)
#
# Comment: That's 10 times faster than crude() method! Could probably be 
#          improved even furher by implementing it as an iterator.

"""
Rejection method
"""
function rejection()
    while true
        u₁ = convert(Int, ceil(rand()*length(pᵢₛ)))
        u₂ = maximum(pᵢₛ)*rand()
        u₂ < pᵢₛ[u₁] && return pᵢₛ[u₁]
    end
end

# julia> simulation = [rejection() for i = 1:10_000]
# julia> histogram(simulation)
#             ┌────────────────────────────────────────┐
# (0.06,0.08] │▇▇▇▇▇▇▇ 650                             │ (pᵢ = 1/16)
#  (0.1,0.12] │▇▇▇▇▇▇▇▇▇▇▇▇ 1062                       │ (pᵢ = 5/48)
# (0.12,0.14] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 1302                     │ (pᵢ = 1/8)
# (0.14,0.16] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 1469                   │ (pᵢ = 7/48)
# (0.24,0.26] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 2430        │ (pᵢ = 1/4)
#  (0.3,0.32] │▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 3087 │ (pᵢ = 5/16)
#             └────────────────────────────────────────┘
#
# Benchmark:
# julia> @time [rejection() for i = 1:10_000]
#   0.038120 seconds (267.05 k allocations: 6.618 MB) 
# Comment: that's 2 times faster than crude() method.

# Results:
# 
#   All three methods were implemented, and for a given discrete distribution 
#   and n = 10000, we have obtained following results:
#
#     crude method:      0.059389 seconds (359.96 k allocations: 12.352 MB)      
#     rejection method:  0.038120 seconds (267.05 k allocations: 6.618 MB) 
#     alias method:      0.006826 seconds (109.04 k allocations: 2.352 MB)
#
#   Which matched our expectation.

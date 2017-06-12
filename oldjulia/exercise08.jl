using Distributions

# ⋅ First do exercise 13 in Chapter 7 of Ross:
# 
#   Let X₁,...,Xₙ be independent identically distributed random variables
#   having uknown mean μ. For given constants a < b, we are interested in
#   esimating p = P{a < Σⁿᵢ₌₁ Xᵢ/n - μ < b}
#
#   a) Explain how we can use the boostrap approach to esimate p.
#      
#      First, we will calculate the mean μ = (Σⁿᵢ₌₁ Xᵢ)/n which we will 
#      assume is our most precise mean (as we are using all the data poi-
#      nts to estimate it). Then we will use the Boostrap method to gene-
#      rate a lot of new versions of the term Σⁿᵢ₌₁ Xᵢ/n substract that 
#      mean from those. Finally we will see what proportion of the results
#      is between a and b - this will be the probability we are looking for.
# 
#   b) Estimate p if n = 10 and the values of the Xᵢ are 56, 101, 78, 67,
#      93, 87, 64, 72, 80, and 69. Take a = -5, b = 5.

# Draw with replacement function used later.
drawwr(X) = [X[rand(1:length(X))] for _ = 1:length(X)]

X = [56, 101, 78, 67, 93, 87, 64, 72, 80, 69]
μ = mean(X)
samples = [mean(drawwr(X)) for _ in 1:100_000].-μ
trues = map(e -> e < 5 && e > -5, samples)
p = sum(trues) / length(samples) # ~ 0.760

# ⋅ Write a subroutine that takes as input a `data` vector of observed 
#   values, and which outputs the median as well as the bootstrap 
#   estimate of the variance of the median, based on r = 100 bootstrap replicates.

function subroutine(data; r = 100)
    many_data    = [drawwr(data) for _ = 1:r]
    many_medians = [median(X) for X in many_data]

    Dict(
        :mean              => mean(data), 
        :median            => median(data), 
        :est_var_of_median => var(many_medians)
    )
end

# ⋅ Test the method: Simulate N = 200 Pareto distributed random variates with 
#   β = 1 and k = 1.05. Compute the mean, the median, and the bootstrap 
#   estimate of the variance of the sample median.

α = 1.05
θ = 1

data = quantile(Pareto(α, θ), rand(200))
subroutine(data)

# ⋅ Compare the precision of the estimated median with the precision of the estimated mean.

nothing

include("common.jl")

# Exercise 5: Variance reduction methods
#
# • Estimate the integral ∫₀¹ exp(x) dx by simulation (the crude Monte
#   Carlo estimator). Use eg. an estimator based on 100 samples and 
#   present the result as the point estimator and a confidence interval.
#
# • Estimate the integral ∫₀¹ exp(x) dx using antithetic variables, 
#   with comparable computer ressources.
#
# • Estimate the integral ∫₀¹ exp(x) dx using a control variable, 
#   with comparable computer ressources.
#
# • Estimate the integral ∫₀¹ exp(x) dx using stratified sampling, 
#   with comparable computer ressources.
#
# • Optional: Use control variates to reduce the variance of the estimator 
#             in exercise 4 (Poisson arrivals).
#
#
# julia> x = linspace(0,2,1000); lineplot(x, exp(x), ylim = [0, 5], xlim=[0, 2])
#      ┌──────────────────────────────────────────────────┐
#    5 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡜⠁⠀⠀⠀⠀⠀⠀⠀│
#      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡼⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠖⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠖⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠖⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#      │⠀⠀⠀⠀⠀⢀⣠⠤⠖⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#      │⣀⣠⠤⠖⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#      │⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#    0 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#      └──────────────────────────────────────────────────┘
#      0                        1                         2
#
#
# • Estimate the integral ∫₀¹ exp(x) dx by simulation (the crude Monte
#   Carlo estimator). Use eg. an estimator based on 100 samples and 
#   present the result as the point estimator and a confidence interval.
#
# julia> n  = 100;
# julia> X  = [e^rand() for i = 1:n];
# julia> μ  = mean(X)
# 1.7076722628994312
#
# julia> σ² = mean(X.^2) - mean(X)^2
# 0.24976235771911126
#
#    Now, motivated by central limit theorem, we construct the confidence 
#    intervals:
#
# julia> t_d = TDist(n - 1); α = 0.05;
# julia> (μ + √(σ²/n) * quantile(t_d, α/2), μ + √(σ²/n) * quantile(t_d, (1-α/2)))
# (1.6085085799152925, 1.8068359458835699)
#
# • Estimate the integral ∫₀¹ exp(x) dx using antithetic variables, 
#   with comparable computer ressources.
#
# julia> Y = [(exp(rand()) + e/exp(rand()))/2 for i = 1:n];
# julia> μ = mean(Y)
# 1.7133340205701004
#
# julia> σ² = mean(Y.^2) - mean(Y)^2
# 0.13493602945518735
#
#   Again, we construct the confidence intervals:
#
# julia> (μ + √(σ²/n) * quantile(t_d, α/2), μ + √(σ²/n) * quantile(t_d, (1-α/2)))
# (1.6622820657544337,1.808057087409888)
#
# • Estimate the integral ∫₀¹ exp(x) dx using a control variable, 
#   with comparable computer ressources.
#
# julia> c = -0.14086 * 12;
# julia> Uₛ= rand(n); 
# julia> Z = exp(Uₛ) + c*(Uₛ - .5);
# julia> μ = mean(Z)
# 1.7148015195590884
#
# julia> σ² = mean(Z.^2) - mean(Z)^2
# 0.0036891532137888206
#
#   Confidence intervals:
#
# julia> (μ + √(σ²/n) * quantile(t_d, α/2), μ + √(σ²/n) * quantile(t_d, (1-α/2)))
# (1.7027497033037926,1.7268533358143843)
#
# • Estimate the integral ∫₀¹ exp(x) dx using stratified sampling, 
#   with comparable computer ressources.
#

# TODO REWRITE IN JULIA TO USE BROADCAST

k = 10

# julia> a = rand(2,1); A = rand(2,3);

# julia> a
# 2×1 Array{Float64,2}:
#  0.951746
#  0.492025

# julia> A
# 2×3 Array{Float64,2}:
#  0.737162  0.0574485  0.336656
#  0.795871  0.198941   0.2526

# julia> broadcast(+, a, A)
# 2×3 Array{Float64,2}:
#  1.68891  1.00919   1.2884
#  1.2879   0.690966  0.744626

# julia>

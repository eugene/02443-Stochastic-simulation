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
# julia> σ² = (1/(n-1))*sum((X.-μ).^2)
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
# julia> σ² = (1/(n-1))*sum((Y.-μ).^2)
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
# julia> σ² = (1/(n-1))*sum((Z.-μ).^2)
# 0.003761952060192394
#
#   Confidence intervals:
#
# julia> (μ + √(σ²/n) * quantile(t_d, α/2), μ + √(σ²/n) * quantile(t_d, (1-α/2)))
# (1.7027497033037926,1.7268533358143843)
#
# • Estimate the integral ∫₀¹ exp(x) dx using stratified sampling, 
#   with comparable computer ressources.
#
# julia> Wᵢ = sum([exp((i/100)+rand()/100) for i = 0:99])/100

strts = n
W = [exp((i/strts)+rand()/strts) for i = 0:strts-1]
μ = mean(Z)
σ² = (1/(n-1))*sum((W.-μ).^2)
(μ + √(σ²/n) * quantile(t_d, α/2), μ + √(σ²/n) * quantile(t_d, (1-α/2)))
(1.618864155240671,1.8150218560345295)

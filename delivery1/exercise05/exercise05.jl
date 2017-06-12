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

"""
We compute ∫₀¹ exp(x) by the taking the mean `n` random numbers drawn from the
distribution `e^U` with U uniform. The expected value of this distribution is
∫₀¹ exp(x).
"""
function mc_crude(n::Int)
    x = exp(rand(n))
    (mean(x), var(x))
end

"""
We compute ∫₀¹ exp(x) by exploiting correlation. Let Y be (e^U + e^(1-U))/2,
then (with X=e^U):
    E[Y] = (E[e^U] + E[e^(1-U)])/2 = E[X]
The variance of Y can be shown to be lower by half of the covariance between U
and 1-U than the variance of X, so the approximation is more accurate.
"""
function mc_antithetic(n::Int)
    x = exp(rand(n));
    y = .5 * (x + e./x);
    (mean(y), var(y))
end

"""
We choose a clever linear combination of two variables that minimize the
variance with respect to X=e^U. Take
    Z = X + c(Y - E[Y])
then, for well-chosen values of c and Y, the variance decreases. Note that the
expected value remains the same, because
    E[c(Y-E[Y])] = c*(E[Y]-E[E[Y]]) = 0
"""
function mc_control_variates(n::Int)
    c = -0.14085 * 12 # empirically determined by kind people from the past
    u = rand(n)
    z = exp(u) + c * (u - .5)
    (mean(z), var(z))
end

"""
We split up the domain [0,1] into `k` strata and sample an equal number  of
times in each stratum. The random variable we consider is then:
W = (e^(0/k + U1/k) + e^(1/k + U2/k) + ⋯ + e^((k-1)/k + Uk/k)/k
"""
function mc_stratisfied_sampling(n::Int, k::Int)
    x = broadcast(+, 0:k-1, rand(k, n)) ./ k
    w = sum(exp(x), 1) ./ k
    (mean(w), var(w))
end

"""
Memory optimized implementation mc_stratisfied_sampling that doesn't
unnecessarily store a very large matrix and avoids allocations by reusing the
same buffer each iteration.

It doesn't seem to matter much.
"""
function mc_stratisfied_sampling_optimized(n::Int, k::Int)
    buffer = zeros(k)
    acc_mean = 0.0
    acc_var = 0.0
    for i in 1:n
        rand!(buffer)
        buffer += 0:k-1
        buffer /= k
        map!(exp, buffer)
        a = sum(buffer) / k
        acc_mean += a
        acc_var += a*a
    end
    (acc_mean/n, acc_var/n - (acc_mean/n)^2)
end

###############################################################################
#                                 EXPERIMENTS                                 #
###############################################################################

"""
Experiment with the above functions. The estimated value, confidence intervals,
and errors are printed.

- Results:
    julia> experiment()
    Confidence intervals are printed as [lower < estimated value < upper]
    Crude:            [1.7042 < 1.7138 < 1.7235]; err=0.004462, order=2.350488
    Antithetic:       [1.7162 < 1.7174 < 1.7186]; err=0.000912, order=3.040136
    Control variates: [1.7171 < 1.7184 < 1.7196]; err=0.000094, order=4.027029
    Strat. sampling:  [1.7183 < 1.7183 < 1.7183]; err=0.000004, order=5.432445
"""
function experiment()
    println("Confidence intervals are printed as [lower < estimated value < upper]")

    n = 10_000
    m, S = mc_crude(n)
    lo, hi = confidence_interval(n, m, S)
    err = abs(m-e+1)
    order = -log10(err)
    @printf("Crude:            [%.4f < %.4f < %.4f]; err=%f, order=%f\n", lo, m, hi, err, order)

    n = 10_000
    m, S = mc_antithetic(n)
    lo, hi = confidence_interval(n, m, S)
    err = abs(m-e+1)
    order = -log10(err)
    @printf("Antithetic:       [%.4f < %.4f < %.4f]; err=%f, order=%f\n", lo, m, hi, err, order)

    n = 10_000
    m, S = mc_control_variates(n)
    lo, hi = confidence_interval(n, m, S)
    err = abs(m-e+1)
    order = -log10(err)
    @printf("Control variates: [%.4f < %.4f < %.4f]; err=%f, order=%f\n", lo, m, hi, err, order)

    n = 10
    strata = 1000
    m, S = mc_stratisfied_sampling(n, strata)
    lo, hi = confidence_interval(n, m, S)
    err = abs(m-e+1)
    order = -log10(err)
    @printf("Strat. sampling:  [%.4f < %.4f < %.4f]; err=%f, order=%f\n", lo, m, hi, err, order)
end

"""
We use stratisfied sampling and try to find out the maximum precision. The
precision seems to linearly increase in the number of strata, given a constant
number of iterations (here 100). The max precision seems to be around 10
significant numbers.

- Results:
    julia> errors, order = experiment_max_precision(20)
    Run 1 : [strata=4      ]; err=0.0100954876838872, order=1.995873
    Run 2 : [strata=8      ]; err=0.0038972099771357, order=2.409246
    Run 3 : [strata=16     ]; err=0.0012915135766618, order=2.888901
    Run 4 : [strata=32     ]; err=0.0000608472772794, order=4.215759
    Run 5 : [strata=64     ]; err=0.0001243335768626, order=3.905412
    Run 6 : [strata=128    ]; err=0.0000190587640969, order=4.719905
    Run 7 : [strata=256    ]; err=0.0000018626747140, order=5.729863
    Run 8 : [strata=512    ]; err=0.0000024835758068, order=5.604923
    Run 9 : [strata=1024   ]; err=0.0000034760328278, order=5.458916
    Run 10: [strata=2048   ]; err=0.0000006970523974, order=6.156735
    Run 11: [strata=4096   ]; err=0.0000001163256205, order=6.934325
    Run 12: [strata=8192   ]; err=0.0000001119147450, order=6.951113
    Run 13: [strata=16384  ]; err=0.0000000218653156, order=7.660244
    Run 14: [strata=32768  ]; err=0.0000000126479958, order=7.897978
    Run 15: [strata=65536  ]; err=0.0000000019841608, order=8.702423
    Run 16: [strata=131072 ]; err=0.0000000005827550, order=9.234514
    Run 17: [strata=262144 ]; err=0.0000000002214933, order=9.654639
    Run 18: [strata=524288 ]; err=0.0000000000193530, order=10.713253
    Run 19: [strata=1048576]; err=0.0000000000972005, order=10.012332
    Run 20: [strata=2097152]; err=0.0000000000164737, order=10.783209

    julia> barplot(map(i->string(2^i), 1:20), order)
               ┌────────────────────────────────────────┐ 
             2 │▪▪▪▪ 1.9958726966999978                 │ 
             4 │▪▪▪▪ 2.4092461942959247                 │ 
             8 │▪▪▪▪▪ 2.8889010240719193                │ 
            16 │▪▪▪▪▪▪▪▪ 4.2157588502850105             │ 
            32 │▪▪▪▪▪▪▪ 3.9054115722677416              │ 
            64 │▪▪▪▪▪▪▪▪▪ 4.719905265465457             │ 
           128 │▪▪▪▪▪▪▪▪▪▪▪ 5.729862980980341           │ 
           256 │▪▪▪▪▪▪▪▪▪▪ 5.6049225793902915           │ 
           512 │▪▪▪▪▪▪▪▪▪▪ 5.458916130708892            │ 
          1024 │▪▪▪▪▪▪▪▪▪▪▪ 6.156734574802675           │ 
          2048 │▪▪▪▪▪▪▪▪▪▪▪▪▪ 6.934324622304229         │ 
          4096 │▪▪▪▪▪▪▪▪▪▪▪▪▪ 6.951112690398281         │ 
          8192 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 7.660244250577429        │ 
         16384 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 7.897978288601655       │ 
         32768 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 8.702423126941506      │ 
         65536 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 9.234514024808833     │ 
        131072 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 9.654639468721339    │ 
        262144 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 10.713252518521063 │ 
        524288 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 10.012331635563704  │ 
       1048576 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 10.783208549671228 │ 
               └────────────────────────────────────────┘ 


    julia> lineplot(1:20, order2)
               ┌────────────────────────────────────────┐ 
            11 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⢆⠀⢀⠔│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁⠀⠑⠃⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠊⠁⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⠀⢀⠔⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠊⠀⠈⠉⠑⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⡰⠑⠢⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⡰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⡠⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠔⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
             1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               └────────────────────────────────────────┘ 
               0                                       20


    julia> lineplot(1:20, errors2)
               ┌────────────────────────────────────────┐ 
          0.02 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠸⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
               │⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
             0 │⠀⠀⠀⠀⠀⠀⠑⠢⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀│ 
               └────────────────────────────────────────┘ 
               0                                       20
"""
function experiment_max_precision(runs=10)
    errors = zeros(runs)
    orders = zeros(runs)

    for i =1:runs
        strata = 2 << i

        n = 100;
        m, S = mc_stratisfied_sampling_optimized(n, strata)
        err = abs(m-e+1)
        order = -log10(err)
        @printf("Run %-2d: [strata=%-7d]; err=%.16f, order=%f\n", i, strata, err, order)

        errors[i] = err
        orders[i] = order
    end

    errors, orders
end

using UnicodePlots
using StatsBase
using Distributions
using UnicodePlots
using GR

runs = 1_000_000

λ = 1
s = 8
n = 10
i = 0:n
A = λ * s

# first, we construct the transition matrix H
H = begin
    H = zeros(11,11)

    # fix the probabilities
    for i in 1:n+1, j in 1:n+1
       (i + j) <= n+2 && (H[i,j] = 1.0)
    end

    for i = 1:n+1
        s = sum(H[i, :])
        H[i, :] = map(e -> (e == 1.0) && (1/s), H[i, :])
    end

    H
end

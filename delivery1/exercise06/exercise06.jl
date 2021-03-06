srand(0)

using UnicodePlots
using StatsBase
using Distributions
using UnicodePlots

runs = 1_000_000

λ = 1
s = 8
n = 10
i = 0:n
A = λ * s

result = A.^i ./ [factorial(iii) for iii ∈ i]
result = result/sum(result)

# julia> lineplot(result)
#        ┌──────────────────────────────────────────────────┐
#    0.2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠉⠉⠉⠉⠑⠤⡀⠀⠀⠀⠀⠀│
#        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⡄⠀⠀⠀│
#        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠢⡀⠀│
#        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢢│
#        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#        │⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#      0 │⣀⣀⣀⣀⡠⠔⠒⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
#        └──────────────────────────────────────────────────┘
#        1                                                 11
#
# julia> barplot(collect(0:10), result)
#       ┌────────────────────────────────────────┐
#     0 │ 0.00041116370815887153                 │
#     1 │ 0.0032893096652709722                  │
#     2 │▪ 0.013157238661083889                  │
#     3 │▪▪▪▪ 0.03508596976289037                │
#     4 │▪▪▪▪▪▪▪▪ 0.07017193952578074            │
#     5 │▪▪▪▪▪▪▪▪▪▪▪▪ 0.11227510324124919        │
#     6 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.14970013765499893   │
#     7 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.17108587160571304 │
#     8 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.17108587160571304 │
#     9 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.1520763303161894    │
#    10 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.12166106425295149      │
#       └────────────────────────────────────────┘

# Generate values from this distribution by applying the Metropolis-Hastings algorithm, 
# verify with a χ2-test. You can use the parameter values from exercise 4.

H    = zeros(11,11)+(1/11)
g(i) = A^i / factorial(i)

function sim()
    X = [rand(0:10)] # initial state

    for _ in 1:runs
        x = X[end]

        # Sample a random variable y, which takes 
        # values in {0..n} according to the distribution
        # given by the row `init_state` of H (H[init_state + 1, :])

        row = H[x + 1, :]
        uᵢ  = rand()
        y   = (findfirst(e -> uᵢ < e, cumsum(row))) - 1
        MH  = min(1, (g(y)*(H[y+1, x+1])) / (g(x)*(H[x+1, y+1])))

        if rand() < MH
            push!(X, y)
        else
            push!(X, x)
        end
    end

    X
end

# julia> barplot(collect(i), counts(sim()))
#       ┌────────────────────────────────────────┐
#     0 │ 358                                    │
#     1 │▪ 3329                                  │
#     2 │▪▪ 13183                                │
#     3 │▪▪▪▪▪▪▪ 35199                           │
#     4 │▪▪▪▪▪▪▪▪▪▪▪▪▪ 69523                     │
#     5 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 112577            │
#     6 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 150031     │
#     7 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 171599 │
#     8 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 171891 │
#     9 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 151073     │
#    10 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 121238          │
#       └────────────────────────────────────────┘

function chisq(X)
    expected = result * runs
    observed = counts(X)

    T_val    = sum(((observed - expected).^2) ./ expected)
    println("T_val: $T_val")

    df   = length(i) - 1
    high = quantile(Chisq(df), 0.95) 

    return T_val <= high
end

# Note:
#
# The simulated results did not pass the Chi² test half of the time,
# but this is expected as Chi² test assumes variable independence.
#

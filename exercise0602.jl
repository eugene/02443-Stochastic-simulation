# Exercise 6 continued
# 
# ⋅ For two diﬀerent call types the joint number of occupied lines is given by
# 
#                      P(i, j) = (1/K) * (A₁ⁱ/i!) * (A₂ʲ/j!)
#
# ⋅ Use Metropolis-Hastings, directly and coordinat wise to generate variates 
# from this distribution. You can use A 1 , A 2 = 4 og n = 10.
# 
# ⋅ Test the distribution with a Χ² test
#
#
# Please see exercise0602-analytical.mov that shows the expected join distribution
# and exercise0602-training.mov that shows the training progress.

using UnicodePlots
using StatsBase
using Distributions
using UnicodePlots
using GR

runs = 100_000

n = 10
i = 0:n
A₁, A₂ = 4, 4
g(i, j) = ((A₁^i) * (A₂^j)) / (factorial(i) * factorial(j))
z = [g(i, j) for i = 1:11, j = 1:11]

# (See output in exercise0602.mov)
function plot_g()
    x = linspace(0, 11, 11)
    y = linspace(0, 11, 11)
    
    # beginprint("anim.mov")
    f = 0; 
    # while f <= ((1/24)*90)
        f += 0.01
        clearws()
        setwindow(0, 11, 0, 11)
        setspace(0, 120, Int(round(45*(1+sin(f)))), Int(round(25*(1+sin(f)))))
        setcharheight(10.0/500)
        axes3d(1, 0, 10, 0,  0, 0, 2, 0, 1, -0.005)
        axes3d(0, 1,  0, 11, 0, 0, 0, 2, 0,  0.005)
        titles3d("i", "j", "g\\(i,j\\)")
        
        surface(x, y, z, 4)
        # surface(x, y, z, 2)
        updatews()
    # end
    # endprint()
end

function sim()
    X, Y = [0], [0]   # initial states
    F = zeros(11, 11) # final

    beginprint("anim.mov")
    for runᵢ in 1:runs
        x, new_x = X[end], 0
        y, new_y = Y[end], 0
    
        new_x   = rand(0:10)
        new_y   = rand(0:10)
        
        gg = g(new_x, new_y) / g(x, y)
        MH  = min(1, gg)

        if rand() < MH
            push!(X, new_x)
            push!(Y, new_y)

            F[new_y+1, new_x+1] = F[new_y+1, new_x+1] +=1
        else
            push!(X, x)
            push!(Y, y)

            F[x+1, y+1] = F[x+1, y+1] +=1
        end

        (runᵢ < 100_000 && (runᵢ % 100 == 0)) && surface(F)
    end
    endprint()

    (F, X, Y)
end

(F, X, Y) = sim()

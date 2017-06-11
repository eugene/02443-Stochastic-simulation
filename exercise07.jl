# Exercise 7
#    - Implement simulated annealing for the travelling salesman.
#    - Have input be positions in plane of the n stations.
#    - Let the cost of going i→j be the Euclidian distance between station i
#      and j.
#    - As cooling scheme, use e.g.  T_k = 1 / √ 1 + k.
#    - The route must end where it started.
#    - Initialise with a random permutation of stations.
#    - As proposal, permute two random stations on the route.
#    - Plot the resulting route in the plane.
#    - Debug first with stations on a circle. Then apply to stations from
#      CampusNet Filesharing.

"""
Solve the traveling salesman problem using simulated annealing. We use λ_n =
log(1+n) as the cooling scheme, as explained in Ros p238.

The argument is the symmetric distance matrix D which contains the Euclidian
distances between any two points. Given coordinates in vectors x and y, this
matrix can be computed with the `Dmatrix` function.
"""
function tsp(D::Matrix{Float32})::Tuple{Float32, Vector{Int}}
    N = size(D, 1)   # should be a square matrix
    x = shuffle(1:N) # random initial permutation
    y = copy(x)

    cost_x = cost(N, D, x);

    for n = 1:100_000
        i,j = perm(N)
        y[i]=x[j]; y[j]=x[i];
        cost_y = cost(N, D, y);

        if cost_y <= cost_x          # if cost of alternative y is better than x, choose y
            cost_x = cost_y;
            x[i]=y[i]; x[j]=y[j];
        else                         # else, choose y anyway with probability p
            p = n^(cost_x-cost_y)    # here, cooling scheme is log(1+n), which causes p to decrease over time

            if rand() < p            # choose y anyway
                cost_x = cost_y;
                x[i]=y[i]; x[j]=y[j];
            else                     # skip y
                y[i]=x[i]; y[j]=x[j];
            end
        end
    end

    cost_x, x
end

"""
A version that computes the same problem `nb_runs` times. If there are
multiple threads available (i.e. Julia is started with a -p <n> or -p <auto>
option), then this will actually run in separate threads.

The reported result is the one with the lowest cost.

Julia will stall when started with multiple threads with GR stuff. Uncomment
the GR using clause and the calls to its methods in order to use multiple
threads. I haven't figured out proper initialization of workers yet.

- Results (with 4 cores / 8 threads):
julia> tsp_random(30)
run 1: 5.978453
run 2: 5.532976
       ⋮
run 15: 5.9445467
run 16: 5.165679
min, max, std.var. = 4.96, 6.27, 0.44
  0.202130 seconds (3.10 k allocations: 179.641 KB)
Initial distance: 15.537192
Approx. min. distance: 4.9568043
Order: [6,20,19,3,12,29,16,9,17,21,26,11,4,8,2,15,28,7,23,24,1,25,5,27,18,10,13,14,30,22]

- Results (with 1 core / 1 thread)
julia> tsp_random(30)
run 1: 6.0724974
run 2: 6.120127
       ⋮
run 15: 6.566638
run 16: 6.268954
min, max, std.var. = 5.02, 6.85, 0.50
  0.867939 seconds (1.02 k allocations: 58.766 KB)
Initial distance: 15.174479
Approx. min. distance: 5.018701
Order: [18,13,19,9,8,16,1,23,17,6,29,24,28,4,27,3,22,20,21,11,12,15,7,30,10,5,2,14,25,26]

"""
function tsp_parallel(D::Matrix{Float32}, nb_runs::Int)::Tuple{Float32,Vector{Int}}
    threads = []
    results = []

    for i in 1:nb_runs
        push!(threads, @spawn tsp(D))
    end

    imin, vmin, xmin = 0, Inf32, Int[]
    for i in 1:nb_runs
        v, x = fetch(threads[i])
        push!(results, (v,x))
        println("run $(i): ", v)
    end

    vs::Vector{Float32} = map(t->t[1], results)
    imin = indmin(vs)
    sigma2 = var(vs)

    @printf("min, max, std.var. = %.2f, %.2f, %.2f\n", minimum(vs), maximum(vs), sqrt(var(vs)))

    results[imin]
end

"""
Compute the cost of a certain order of nodes. `N` is the number of nodes, `D`
is the distance matrix and `x` is a permutation of 1:N.

The distance includes the distance from the last node back to the first node.
"""
function cost(N::Int, D::Matrix{Float32}, x::Vector{Int})::Float32
    cost = 0.0f0
    for i in 1:N
        cost += D[x[i], x[i%N+1]]
    end
    cost
end

"""
Generate two numbers `i` and `j`, `i != j` in the range 1:N.
"""
function perm(N::Int)::Tuple{Int,Int}
    i = rand(1:N)
    j = rand(1:N)
    while j == i
        j = rand(1:N)
    end
    (i,j)
end 

"""
Compute the distance matrix given vectors `x` and `y` containing the x resp.
the y coordinates of the points to visit.
"""
function Dmatrix(x, y)
    N = length(x);
    D = zeros(Float32, N, N);
    for i in 1:N, j in i+1:N
        D[i, j] = sqrt((x[j]-x[i])^2 + (y[j]-y[i])^2)
        D[j, i] = D[i, j]
    end
    D
end

"""
Hardcoded example used to compare implementations.

- Results:
julia> tsp_hardcode1()
Initial distance: 163.9949
Approx. min. distance: 144.294
Order: [5,8,9,7,6,10,12,4,1,2,3,11]
      ┌────────────────────────────────────────┐ 
   30 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠉⠉⠉⠒⠒⠢⠤⠤⢄⣀⣀⠔⠊⠉⠢⣀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁⠀⠀⠀⠀⠀⠀⠀⢀⠔⠊⠀⠉⠉⠉⠒⠒⠳⢤⡀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁⠀⠀⠀⠀⠀⠀⢀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁⠀⠀⠀⠀⠀⢀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁⠀⠀⠀⠀⢀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠀⢀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⢀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⢀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⢀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⢃⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠠⡤⢄⣀⡀⠀⠀⠀⠀⢀⠗⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠑⢄⢀⠾⡉⠒⠒⢔⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⡨⠣⡀⢣⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⡔⠁⠀⢘⠌⣆⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
    0 │⠈⠉⠒⠖⠁⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      └────────────────────────────────────────┘ 
      0                                       60

      ┌────────────────────────────────────────┐ 
   30 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠉⠢⢄⡀⠀⠀⠀⠀⠀⠀⣀⠤⠊⠉⠢⡀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠊⠀⠀⠀⠀⠈⠑⠢⢄⢀⠤⠊⠀⠀⠀⠀⠀⠈⠢⢄⡀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢈⡱│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠊⠁⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠒⠉⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠔⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠔⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⢀⠜⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⡠⠃⠀⠀⠀⠀⠀⢀⡠⠔⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⢠⠤⢄⣀⡀⡔⠁⠀⠀⠀⣀⠤⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⢸⠀⠀⠀⠈⠀⠀⢀⠔⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⢸⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⢸⠀⠀⠀⢀⠤⣠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
    0 │⠈⠉⠒⠖⠁⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      └────────────────────────────────────────┘ 
      0                                       60

"""
function tsp_hardcode1()
    t = [(1,2), (5,1), (8,3), (1,9), (12,7), (41,26), (50,30), (60,25), (55,27), (30,30), (10,2), (7,8)];
    x = map(t -> t[1], t);
    y = map(t -> t[2], t);

    D = Dmatrix(x, y)
    @time v, s = tsp_parallel(D, 16)

    report_result(x, y, s)
end

"""
Hardcoded example from the course web page.

We don't have any graphs for this; we didn't figure out a set of points that
corresponds with the given D.

- Results:
julia> tsp_hardcode2()
Initial distance: 3404.0
Approx. min. distance: 797.0
Order: [12,6,1,13,4,3,20,11,18,7,17,10,16,5,8,2,9,14,19,15]

"""
function tsp_hardcode2()
    v, s = tsp_parallel(Dcourse, 16);
    N = size(Dcourse, 1)
    println("Initial distance: ", cost(N, Dcourse, collect(1:N)))
    println("Approx. min. distance: ", cost(N, Dcourse, s))
    println("Order: ", s)
end

"""
Generate `N` points on the unit circle and shuffle them. Then solve the tsp.

- Results:
julia> tsp_circle(15)
Initial distance: 21.017775
Approx. min. distance: 6.2373514
Order: [2,9,3,10,6,13,14,1,11,8,7,12,5,15,4]
      ┌────────────────────────────────────────┐ 
    1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡠⠤⠔⠒⠊⠉⡆⠀⠀⠀⠀⠀⠀⢠⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠢⣀⠀⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⡰⠁⡇⠀⠀⠀⠀⠀⢀⡀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠢⣀⠀⠀⠀⡇⠀⠀⠀⡰⢁⣀⡧⠤⠔⠒⠊⠉⠁⢸⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠈⠛⢖⡒⠤⠤⣀⣀⠀⠀⠀⠀⠀⣑⣢⣤⠼⡔⠒⡞⠉⠁⠀⡇⠀⠀⠀⠀⠀⠀⠀⢇⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠈⠒⣄⣀⠤⠭⠛⠓⠛⠫⠤⢄⣀⡑⢧⣎⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠸⣀⣀⠤⡆⠀│ 
      │⣀⣀⠤⠤⠒⠒⠉⠉⠀⠈⠒⢄⡀⠀⠀⠀⠀⠀⠀⢨⠛⡕⠓⠢⣤⢄⣇⡀⠀⠀⠀⣀⠤⠔⠊⢣⠀⠀⠸⡀│ 
      │⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠒⢄⡀⠀⠀⡠⠃⠀⢣⠀⠀⠀⠑⣧⡨⠭⠓⠛⠢⠤⠤⣀⣘⡄⠀⠀⢇│ 
      │⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠢⣴⠁⠀⠀⠈⣆⠤⠒⠊⡇⠈⠑⠤⡀⠀⠀⠀⠀⠀⢹⠉⠒⠺│ 
      │⠀⠸⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⣀⡩⠶⢎⠉⢱⠀⠀⠀⡇⠀⠀⠀⠈⠑⠤⡀⠀⠀⠈⡆⠀⠀│ 
      │⠉⠢⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⢤⠞⠉⠀⠀⠀⠀⠉⠢⣇⠀⠀⡇⠀⠀⠀⠀⠀⠀⠈⠑⠤⡀⢱⠀⠀│ 
      │⠀⠀⠘⡯⡢⢄⠀⠀⣀⠤⠔⠊⠁⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠸⡉⠢⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⠇⠀│ 
      │⠀⠀⠀⢱⡨⠶⡓⠫⣀⠀⠀⠀⢠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢇⠀⡇⠉⠢⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠈⠢⡀⠑⠢⣰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⡀⡇⠀⠀⠀⠉⠢⢄⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠀⠈⢢⡔⠁⠑⠢⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣⣇⡠⠤⠔⠒⠒⠉⠁⠀⠀⠀⠀⠀⠀│ 
   -1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠢⣀⣀⡠⠤⠔⠒⠊⠉⠙⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      └────────────────────────────────────────┘ 
      -1                                       1

      ┌────────────────────────────────────────┐ 
    1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡠⠤⠔⠒⠊⠉⠉⠉⠉⠉⠉⠒⠒⠒⠒⠤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⡠⠔⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠑⠢⢄⡀⠀⠀⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⣀⠔⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠢⡀⠀⠀⠀⠀│ 
      │⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠢⡀⠀⠀│ 
      │⠀⢠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢆⠀│ 
      │⡠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⡀│ 
      │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣│ 
      │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
      │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜│ 
      │⠙⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃│ 
      │⠀⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠎⠀│ 
      │⠀⠀⠈⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠁⠀⠀│ 
      │⠀⠀⠀⠀⠑⠢⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠁⠀⠀⠀⠀│ 
      │⠀⠀⠀⠀⠀⠀⠀⠑⠢⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠔⠊⠁⠀⠀⠀⠀⠀⠀│ 
   -1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠒⠒⠤⠤⣀⣀⣀⣀⣀⣀⣀⡠⠤⠤⠔⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
      └────────────────────────────────────────┘ 
      -1                                       1

"""
function tsp_circle(N)
    θ = shuffle!(map(i -> (i*2*π)/N, 0:N-1))
    x = cos(θ)
    y = sin(θ)

    D = Dmatrix(x, y)
    @time v, s = tsp_parallel(D, 16)

    report_result(x, y, s)
end

"""
Generate `N` random points in the unit square and solve the tsp.

- Results:
julia> tsp_random(15)
Initial distance: 7.216254
Approx. min. distance: 3.7522388
Order: [5,7,1,9,10,2,6,14,4,15,3,11,13,12,8]
     ┌────────────────────────────────────────┐ 
   1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⡱⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠊⠀⠀⠀⠀⣀⣀⡠⣤⡤⠴⠀│ 
     │⠀⠀⣀⡀⠀⠀⠀⠀⢀⠔⠁⡰⠁⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⠴⠮⠤⠒⠒⠒⢉⡩⠭⠒⠊⠉⠀⠀⠀⠀│ 
     │⠀⠀⠈⠪⡉⠉⢒⠖⠥⠤⣴⣁⡠⠤⠤⠔⠒⠒⠊⠉⢉⡤⠚⠁⠀⣀⠤⠒⠒⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠑⣤⠊⠀⠀⡰⠁⠀⠀⠀⠀⠀⠀⢀⡠⠞⢁⡠⠔⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⡠⠊⠀⠑⢄⡰⠁⠀⠀⠀⠀⢀⣠⣖⠭⠒⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠈⡆⠀⠀⠀⡰⠣⡀⠀⢀⣠⠖⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⢣⠀⠀⡰⠁⠀⢈⠦⡋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠘⡄⡰⠁⡠⠒⠁⠀⠘⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⣷⣕⣉⣀⣀⣀⣀⣀⠀⠑⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⢸⢇⠀⠀⠀⠀⠀⠀⠉⠉⠒⠳⡢⠤⢄⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⠤⠤⢔⡲⠞⠋⠁│ 
     │⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠢⡀⠀⠀⠉⠉⠒⠒⠤⠔⠒⠊⠉⠉⢀⡠⠔⠊⠁⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⢹⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢢⠀⠀⠀⠀⠀⠀⠀⣀⠤⠒⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠘⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢄⠀⣀⠤⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   0 │⠀⠀⠀⠀⠀⠀⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     └────────────────────────────────────────┘ 
     0                                      0.9

     ┌────────────────────────────────────────┐ 
   1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠓⠢⠤⣀⡀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠈⠑⠒⠤⣀⡀⠀⠀⠀⠀⠀⠀⠀⡰⠁⠀⠀⠀⠀⠈⠉⠑⢲⠀│ 
     │⠀⠀⣀⡀⠀⠀⠀⠀⠀⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⠒⠤⣀⡀⠀⡰⠁⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀│ 
     │⠀⠀⡇⠈⠉⠉⠒⠒⠤⠤⢄⡜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀│ 
     │⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀│ 
     │⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡄│ 
     │⠀⠀⠈⠉⠑⠒⠢⠤⠤⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇│ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇│ 
     │⠀⠀⠀⠀⡠⠤⠤⠤⠤⣀⣀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇│ 
     │⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⠤⠤⠔⠒⠊⠉⠁│ 
     │⠀⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠔⠒⠊⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     │⠀⠀⠀⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   0 │⠀⠀⠀⠀⠀⠀⠓⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠉⠉⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
     └────────────────────────────────────────┘ 
     0                                      0.9

"""
function tsp_random(N)
    x, y = rand(N), rand(N)
    D = Dmatrix(x, y)
    @time v, s = tsp_parallel(D, 16)

    report_result(x, y, s)
end

#using GR
using UnicodePlots
function report_result(x, y, s)
    N = length(x)
    D = Dmatrix(x, y)
    println("Initial distance: ", cost(N, D, collect(1:N)))
    println("Approx. min. distance: ", cost(N, D, s))
    println("Order: ", s)

    xi = push!(copy(x), x[1])
    yi = push!(copy(y), y[1])
    xf = push!(x[s], x[s][1])
    yf = push!(y[s], y[s][1])

    #GR.clearws()
    #GR.subplot(1,2,1)
    #GR.plot(xi, yi)
    #GR.subplot(1,2,2)
    #GR.plot(xf, yf)
    #GR.updatews()

    plot1 = lineplot(xi, yi, grid=false)
    plot2 = lineplot(xf, yf, grid=false)

    println(plot1);
    println(plot2);
end

# Distance array provided on course website
Dcourse = Float32[
           0   225  110    8  257   22   83  231  277  243   94   30    4  265  274  250   87   83  271   86;
         255     0  265  248  103  280  236   91    3   87  274  265  236    8   24   95  247  259   28  259;
          87   236    0   95  248  110   25  274  250  271    9  244   83  250  248  280   29   26  239    7;
           8   280   83    0  236   28   91  239  280  259  103   23    6  280  244  259   95   87  230   84;
         268    87  239  271    0  244  275    9   84   25  244  239  275   83  110   24  274  280   84  274;
          21   265   99   29  259    0   99  230  265  271   87    5   22  239  236  250   87   95  271   91;
          95   236   28   91  247   93    0  247  259  244   27   91   87  268  275  280    7    8  240   27;
         280    83  250  261    4  239  230    0  103   24  239  261  271   95   87   21  274  255  110  280;
         247     9  280  274   84  255  259   99    0   87  255  274  280    3   27   83  259  244   28  274;
         230   103  268  275   23  244  264   28   83    0  268  275  261   91   95    8  277  261   84  247;
          87   239    9  103  261  110   29  255  239  261    0  259   84  239  261  242   24   25  242    5;
          30   255   95   30  247    4   87  274  242  255   99    0   24  280  274  259   91   83  247   91;
           8   261   83    6  255   29  103  261  247  242  110   29    0  261  244  230   87   84  280  100;
         242     8  259  280   99  242  244   99    3   84  280  236  259    0   27   95  274  261   24  268;
         274    22  250  236   83  261  247  103   22   91  250  236  261   25    0  103  255  261    5  247;
         244    91  261  255   28  236  261   29  103    9  242  261  244   87  110    0  242  236   95  259;
          84   236   27   99  230   83    7  259  230  230   22   87   93  250  255  247    0    9  259   24;
          91   242   28   87  250  110    6  271  271  255   27  103   84  250  271  244    5    0  271   29;
         261    24  250  271   84  255  261   87   28  110  250  248  248   22    3  103  271  248    0  236;
         103   271    8   91  255   91   21  271  236  271    7  250   83  247  250  271   22   27  248    0];

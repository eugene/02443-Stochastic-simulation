include("common.jl")

import Base.Collections: enqueue!, dequeue!

# Exercise 4
#
#   • Write a discrete event simulation program for a blocking system, 
#     i.e. a system with n service units and no waiting room.
#   • The arrival process is modelled as a Poisson process.
#   • Choose first the service time distribution as exponential.
#   • Record the fraction of blocked customers, and a confidence 
#     interval for this fraction.
#   • The program should take the offered traffic and the number 
#     of service units as input parameters.
#
# Example: n = 10, mean service time = 8 time units, mean time between 
# customers = 1 time unit (corresponding to an offered traffic of 8 erlang), 
# 10 x 10.000 customers.
#
# In the above example substitute the arrival process with a renewal 
# process with
# 1) Erlang distributed inter arrival times 
# 2) hyper exponential inter arrival times. 
# 
# The Erlang distribution should have a mean of 1, the parameters 
# for the hyper exponential distribution should be
# p1 = 0.8, λ1 = 0.8333, p2 = 0.2, λ2 = 5.0.
#
# • Finally experiment with different service time distributions. 
#   Suggestions are constant service time and Pareto distributed 
#   service times, for Pareto will k = 1.05 and k = 2.05 be interesting 
#   choices. It is recommended that the service time distribution has
#   the same mean (8).
#
# • Make the experiment with a distribution of (your own) choice. 
#   Remember that the distribution should take only non-negative values.

type Event
    event_type::Symbol # :arrive or :service
    time::Float64      # time when this event is scheduled for
end

typealias PQ Collections.PriorityQueue{Event, Float64, Base.Order.ForwardOrdering}

type World{AD <: Distribution, SD <: Distribution}
    arrive_d::AD  # arrive distribution (e.g. Poissson)
    service_d::SD # service distribution (e.g. Exponential)

    event_list::PQ          # event queue (ascending sort)
    available_units::Int    # number of currently available units

    n_blocked::Int          # number of blocked customers
    n_served::Int           # number of served customers
    limit::Int              # max number of customers
end

function World(;n_service_units=10, arrive_d=Poisson(1), service_d=Exponential(8), limit=10_000)
    AD = typeof(arrive_d);
    SD = typeof(service_d);
    World{AD,SD}(arrive_d, service_d, PQ(), n_service_units, 0, 0, limit)
end

function Event(event_type::Symbol, world::World, time_offset)
    target_time::Float64 = 0.0
    if event_type == :arrive
        target_time = Base.rand(world.arrive_d)
    elseif event_type == :service
        target_time = Base.rand(world.service_d)
    else
        error("invalid event type");
    end

    Event(event_type, time_offset + target_time)
end

function handle_arrive(world, event)
    # return if we reached the limit of customers served
    world.n_served+world.n_blocked >= world.limit && return

    # enqueue a new arrive event 
    new_event = Event(:arrive, world, event.time)
    enqueue!(world.event_list, new_event, new_event.time)
    
    # any available service units?
    if world.available_units > 0
        # units are available, let's schedule a service event
        service_event = Event(:service, world, event.time)
        enqueue!(world.event_list, service_event, service_event.time)

        # mark one more unit as busy
        world.available_units -= 1
    else
        # no unit available - we update stats by 
        # incrementing blocked count
        world.n_blocked += 1
    end

    nothing
end

function handle_service(world, event)
    # mark unit as available again and updated statistics
    world.available_units += 1
    world.n_served += 1
end

function simulate(world::World)
    initial_event = Event(:arrive, world, 0.0)
    enqueue!(world.event_list, initial_event, initial_event.time)

    while length(world.event_list) > 0
        event = dequeue!(world.event_list)

        if event.event_type == :arrive
            handle_arrive(world, event)
        elseif event.event_type == :service
            handle_service(world, event)
        else
            error("invalid event type")
        end
    end

    world
end

function runsim(arrive_d, service_d; limit=10_000, runs=10, units=10)
    w = () -> World(n_service_units=units,
                  arrive_d=arrive_d,
                  service_d=service_d,
                  limit=limit)
    @time runs = [simulate(w()) for _ in 1:runs]
    ci = confidence_interval(map(w -> w.n_blocked/(w.n_blocked+w.n_served), runs))

    # print some stats
    for w in runs
        println("served: ", w.n_served, ", blocked: ", w.n_blocked);
    end
    @printf("Confidence: lower < mean < upper = %.4f < %.4f < %.4f\n", ci[1], ci[2], ci[3])
    ci
end

###################################################################################
#                                   EXPERIMENTS                                   #
# The experiments use varying arrival and service distributions. 10_000 clients   #
# are served or blocked in each run, and 10 runs are executed. The number of      #
# served and blocked clients is collected. For each experiment, a confidence      #
# interval of the relative number of blocked clients is provided.                 #
###################################################################################

"""
Run the simulation with the following properties:
    - units:     10
    - arrivals:  Poisson(1)
    - services:  Exponential(8)
    - limit:     10_000
    - α:         0.05
    - runs:      10

- Results:
    julia> run_poisson1_exp8()
      0.092874 seconds (393.18 k allocations: 17.451 MB)
    served: 8699, blocked: 1310
    served: 8690, blocked: 1320
    served: 8722, blocked: 1288
    served: 8654, blocked: 1354
    served: 8792, blocked: 1212
    served: 8754, blocked: 1254
    served: 8662, blocked: 1347
    served: 8735, blocked: 1275
    served: 8781, blocked: 1227
    served: 8767, blocked: 1243
    Confidence: lower < mean < upper = 0.1247 < 0.1282 < 0.1317
"""
run_poisson1_exp8() = runsim(Poisson(1), Exponential(8))

"""
Run the simulation with the following properties:
    - units:     10
    - arrivals:  Poisson(1)
    - services:  Poisson(8)
    - limit:     10_000
    - α:         0.05
    - runs:      10

- Results:
    julia> run_poisson1_poisson8()
      0.099335 seconds (397.15 k allocations: 17.623 MB, 1.94% gc time)
    served: 8988, blocked: 1017
    served: 8951, blocked: 1056
    served: 8888, blocked: 1118
    served: 8930, blocked: 1078
    served: 8964, blocked: 1044
    served: 8898, blocked: 1107
    served: 8913, blocked: 1092
    served: 8891, blocked: 1115
    served: 8944, blocked: 1064
    served: 8941, blocked: 1069
    Confidence: lower < mean < upper = 0.1052 < 0.1075 < 0.1099
"""
run_poisson1_poisson8() = runsim(Poisson(1), Poisson(8))

"""
Run the simulation with the following properties:
    - units:     10
    - arrivals:  Erlang(1,1)
    - services:  Exponential(8)
    - limit:     10_000
    - α:         0.05
    - runs:      10

- Results:
    julia> run_erlang_exp8()
      0.098220 seconds (404.92 k allocations: 17.963 MB)
    served: 9323, blocked: 684
    served: 9403, blocked: 605
    served: 9311, blocked: 697
    served: 9336, blocked: 671
    served: 9271, blocked: 734
    served: 9303, blocked: 703
    served: 9282, blocked: 728
    served: 9323, blocked: 682
    served: 9352, blocked: 657
    served: 9337, blocked: 671
    Confidence: lower < mean < upper = 0.0656 < 0.0683 < 0.0709
"""
run_erlang_exp8() = runsim(Erlang(8,1/8), Exponential(8))

"""
Run the simulation with the following properties:
    - units:     10
    - arrivals:  HyperExponential([.8, .2], [.8333, 5.0])
    - services:  Exponential(8)
    - limit:     10_000
    - α:         0.05
    - runs:      10

- Results:
    julia> run_hyperexp_exp8()
      0.094777 seconds (715.59 k allocations: 24.675 MB, 2.11% gc time)
    served: 9540, blocked: 468
    served: 9446, blocked: 560
    served: 9508, blocked: 494
    served: 9516, blocked: 490
    served: 9573, blocked: 436
    served: 9570, blocked: 440
    served: 9525, blocked: 482
    served: 9492, blocked: 512
    served: 9516, blocked: 493
    served: 9482, blocked: 521
    Confidence: lower < mean < upper = 0.0463 < 0.0489 < 0.0516
"""
run_hyperexp_exp8() = runsim(HyperExponential([.8,.2], [.83333,5.0]), Exponential(8))

# A quick implementation of a HyperExponential distribution with only support
# for random number generation.
type HyperExponential <: Distributions.ContinuousUnivariateDistribution
    ps::Vector{Float64}
    λs::Vector{Float64}
end

function Base.rand(d::HyperExponential)
    # Pick one of the exponential distributions with the respective
    # probabilities in `ps', then generate a random number in that exponential
    # distribution.
    i = Base.rand(Categorical(d.ps))
    Base.rand(Exponential(d.λs[i]))
end

"""
Run the simulation with the following properties:
    - units:     10
    - arrivals:  Poisson(1)
    - services:  Pareto(1.05)
    - limit:     10_000
    - α:         0.05
    - runs:      10

- Results:
    julia> run_poisson1_pareto105()
      0.094365 seconds (387.37 k allocations: 17.169 MB, 2.98% gc time)
    served: 8854, blocked: 1153
    served: 8346, blocked: 1661
    served: 8509, blocked: 1501
    served: 9016, blocked: 987
    served: 8966, blocked: 1040
    served: 7894, blocked: 2115
    served: 7809, blocked: 2199
    served: 8632, blocked: 1377
    served: 8383, blocked: 1623
    served: 8182, blocked: 1824
    Confidence: lower < mean < upper = 0.1246 < 0.1547 < 0.1848
"""
run_poisson1_pareto105() = runsim(Poisson(1), Pareto(1.05))

"""
Run the simulation with the following properties:
    - units:     10
    - arrivals:  Poisson(0.5)
    - services:  Pareto(2.05)
    - limit:     10_000
    - α:         0.05
    - runs:      10

- Results:
    julia> run_poisson05_pareto205()
      0.088927 seconds (416.70 k allocations: 18.614 MB, 2.78% gc time)
    served: 9445, blocked: 557
    served: 9471, blocked: 533
    served: 9428, blocked: 578
    served: 9479, blocked: 523
    served: 9364, blocked: 639
    served: 9468, blocked: 535
    served: 9374, blocked: 632
    served: 9480, blocked: 526
    served: 9444, blocked: 557
    served: 9419, blocked: 584
    Confidence: lower < mean < upper = 0.0536 < 0.0566 < 0.0596
"""
run_poisson05_pareto205() = runsim(Poisson(0.5), Pareto(2.05))

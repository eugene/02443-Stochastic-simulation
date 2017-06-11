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
#   Suggestions are constant service timeand Pareto distributed 
#   service times, for Pareto will k = 1.05 and k = 2.05 be interesting 
#   choices. It is recommended that the service time distribution has
#   the same mean (8).
#
# • Make the experiment with a distribution of (your own) choice. 
#   Remember that the distribution should take only non-negative values.

type Event
    event_type::Symbol # :arrive or :service
    unit::Int64        # what unit services this event (if any)
    time::Float64      # time when this event is scheduled for
end

type World
    # Arrive distribution (Poisson by default)
    arrive_d::Distribution

    # Service distribution (eg. Exponential)
    service_d::Distribution

    # Event list (ascending sort)
    event_list::Collections.PriorityQueue{Event, Float64, Base.Order.ForwardOrdering}

    # this will hold the service units and their availability 
    service_units::Vector{Bool}

    # Number of blocked customers here
    n_blocked::Int64

    # Total number of served customers
    n_served::Int64

    # Some sort of a limit, in this case max number of total customers
    limit::Int64

    function World(; n_service_units = 10, arrive_d = Poisson(1), service_d = Exponential(8), limit = 10_000)
        self = new()
        self.arrive_d = arrive_d
        self.service_d = service_d
        self.event_list = Collections.PriorityQueue{Event, Float64, Base.Order.ForwardOrdering}()
        self.service_units = trues(n_service_units) 
        self.n_blocked = 0
        self.n_served = 0
        self.limit = limit
        self
    end
end

function Event(event_type::Symbol, world::World; time_offset = 0.0, unit = 0)
    target_time = begin
        if event_type == :arrive; quantile(world.arrive_d, rand())
        else                      quantile(world.service_d, rand()) end    
    end

    Event(event_type, unit, time_offset + target_time)
end

function handle_arrive(world, event)
    # return if we reached the limit of customers served
    world.n_served >= world.limit && return

    # enqueue a new arrive event 
    new_event = Event(:arrive, world, time_offset = event.time)
    enqueue!(world.event_list, new_event, new_event.time)
    
    # any available service units?
    unit = findfirst(world.service_units)
    if unit > 0
        # yay! we had a unit available. let's schedule a 
        # service event at that unit 
        service_event = Event(:service, world, time_offset = event.time, unit = unit)
        enqueue!(world.event_list, service_event, service_event.time)

        # and also mark it as being busy
        world.service_units[unit] = false
    else
        # no unit available - we update stats by 
        # incrementing blocked count
        world.n_blocked += 1
    end

    nothing
end

function handle_service(world, event)
    # unit is available again
    world.service_units[event.unit] = true

    # and we update stats
    world.n_served += 1
end

function simulate(world::World)
    initial_event = Event(:arrive, world)
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

    "Served: $(world.n_served), blocked: $(world.n_blocked)"
end

# julia> @time [simulate(World()) for i = 1:10]
#   0.219530 seconds (2.05 M allocations: 61.735 MB, 3.82% gc time)
# 10-element Array{String,1}:
#  "Served: 10004, blocked: 1"
#  "Served: 10001, blocked: 1"
#  "Served: 10001, blocked: 3"
#  "Served: 10000, blocked: 1"
#  "Served: 10002, blocked: 0"
#  "Served: 10001, blocked: 2"
#  "Served: 10002, blocked: 0"
#  "Served: 10002, blocked: 0"
#  "Served: 10000, blocked: 5"
#  "Served: 10003, blocked: 2"

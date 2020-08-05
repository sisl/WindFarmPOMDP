"""
!!! WARNING: Remember to start Julia with `julia -p auto` or `julia -p N` where N is the number of cores chosen !!!

The user should pass in the the following arguments in the Terminal.

# Arguments 
- `script_name` the name of the script to run 
- `no_of_runs` number of episodes to run 
- `tree_queries` number of queries down the tree 
"""

using Distributed
println("### Number of workers: $(nworkers())")

using POMDPs, POMDPModelTools, POMDPSimulators, POMDPPolicies, POMDPModelTools
using BasicPOMCP, ARDESPOT, POMCPOW
using DelimitedFiles

include("../src/windfarmpomdp.jl")
include("../src/beliefstates.jl")
include("../src/windfarm_expertpolicies.jl")

script_name = ARGS[1]
no_of_runs = parse(Int, ARGS[2])
tree_queries_generic = parse(Int, ARGS[3])


dir = string("Generic_Run_Results")
!isdir(dir) ? mkdir(dir) : nothing
dt = Dates.now()

global REWARD_RECORDS = []
<<<<<<< HEAD
# Threads.@threads for c in 1:no_of_runs    # Runs in parallel threads.
for c in 1:no_of_runs                       # Runs series.
=======
Threads.@threads for c in 1:no_of_runs    # Runs in parallel threads.
>>>>>>> 8aa2ac0c7d4e277962b9f10923c4f156c8d650e0
    @show c
    time_taken = @elapsed include(script_name)
    push!(REWARD_RECORDS, sum(rewards_history))

    savename = string("./$dir/", script_id, '_', dt, ".csv")
    writedlm_append(savename, hcat(tree_queries_generic, time_taken, rewards_history', sum(rewards_history)))
end
@show REWARD_RECORDS
@show average(REWARD_RECORDS)

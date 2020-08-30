"""
The user should pass in the the following arguments in the Terminal.

# Arguments 
- `script_name` the name of the script to run 
- `no_of_runs` number of episodes to run 
- `tree_queries` number of queries down the tree 
"""

if Threads.nthreads() == 1
    @warn "You are not running Julia in multiple threads. Aborted.\nRun e.g. `export JULIA_NUM_THREADS=16` in Terminal before running script."
    exit()
end

using POMDPs, POMDPModelTools, POMDPSimulators, POMDPPolicies
using BasicPOMCP, POMCPOW
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
# Threads.@threads for c in 1:no_of_runs    # Runs in parallel threads.
for c in 1:no_of_runs                       # Runs series.
    @show c
    time_taken = @elapsed include(script_name)
    push!(REWARD_RECORDS, sum(rewards_history))

    savename = string("./$dir/", script_id, '_', dt, ".csv")
    writedlm_append(savename, hcat(tree_queries_generic, time_taken, rewards_history', sum(rewards_history)))
end
@show REWARD_RECORDS
@show average(REWARD_RECORDS)

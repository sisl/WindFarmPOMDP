"""
The user should pass in the the following arguments in the Terminal.

# Arguments 
- `script_name` the name of the script to run 
- `no_of_runs` number of episodes to run 
"""

using POMDPs, POMDPModelTools, POMDPSimulators, POMDPPolicies, POMDPModelTools
using BasicPOMCP, ARDESPOT, POMCPOW
using DelimitedFiles

include("../src/windfarmpomdp.jl")
include("../src/beliefstates.jl")
include("../src/windfarm_expertpolicies.jl")

script_name = ARGS[1]
no_of_runs = parse(Int, ARGS[2])

dir = string("Generic_Run_Results")
!isdir(dir) ? mkdir(dir) : nothing
dt = Dates.now()

global REWARD_RECORDS = []
for c in 1:no_of_runs
    @show c
    time_taken = @elapsed include(script_name)
    push!(REWARD_RECORDS, sum(rewards_history))

    savename = string("./$dir/", script_id, '_', dt, ".csv")
    writedlm_append(savename, hcat(time_taken, rewards_history'))
end
@show REWARD_RECORDS
@show average(REWARD_RECORDS)
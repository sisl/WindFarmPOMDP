using POMDPs, POMDPModelTools, POMDPSimulators, POMDPPolicies, POMDPModelTools
using BasicPOMCP, ARDESPOT, POMCPOW
# using D3Trees, ProfileView

include("../src/windfarmpomdp.jl")
include("../src/beliefstates.jl")
include("../src/windfarm_expertpolicy.jl")

# Construct POMDP
no_of_sensors = 5
delta = 220
wfparams = WindFarmBeliefInitializerParams(nx=20,ny=20,grid_dist_obs=220)
pomdp = WindFarmPOMDP(wfparams.nx, wfparams.ny, wfparams.grid_dist, wfparams.altitudes, no_of_sensors, delta)

# Get initial belief distribution (sparse version of GWA data) and initial state
b0 = initialize_belief(wfparams)
s0 = initialize_state(wfparams)

# Construct Belief Updater
up = WindFarmBeliefUpdater(wfparams.grid_dist)

# Define Solver
policy = WindFarmExpertPolicy(pomdp)


println("### Starting Stepthrough ###")
global actions_history = []
global obs_history = []
global rewards_history = []
global belief_history = []
for (s, a, r, o, b, t, bp) in stepthrough(pomdp, policy, up, b0, s0, "s,a,r,o,b,t,bp", max_steps=no_of_sensors)
    # @show s
    @show a
    @show o
    @show r
    @show t
    push!(actions_history, a)
    push!(obs_history, o)
    push!(rewards_history, r)
    push!(belief_history, bp)
end

plot_WindFarmPOMDP_policy!(wfparams, actions_history)

# @time _, info = action_info(planner, b0, tree_in_info=true)
# @time _, info = action_info(planner, b0, tree_in_info=true)
# @profview _, info = action_info(planner, b0, tree_in_info=true)

# @code_warntype _, info = action_info(planner, b0, tree_in_info=true)
# tree = info[:tree]
# inbrowser(D3Tree(tree, init_expand=1), "firefox")

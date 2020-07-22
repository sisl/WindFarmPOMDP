using POMDPs, POMDPModelTools, POMDPSimulators, POMDPPolicies, POMDPModelTools
using BasicPOMCP, ARDESPOT, POMCPOW
# using D3Trees, ProfileView

include("../src/windfarmpomdp.jl")
include("../src/beliefstates.jl")
include("../src/windfarm_expertpolicies.jl")

# Construct POMDP
no_of_sensors = 5
delta = 220 * 4
wfparams = WindFarmBeliefInitializerParams(nx=20,ny=20)
pomdp = WindFarmPOMDP(wfparams.nx, wfparams.ny, wfparams.grid_dist, wfparams.altitudes, no_of_sensors, delta)

# Get initial belief distribution and initial state
b0 = initialize_belief_no_prior(wfparams)
s0 = initialize_state(wfparams)

# Construct Belief Updater
up = WindFarmBeliefUpdater(wfparams.altitudes, wfparams.grid_dist)

# Define Solver
policy = WindFarmGreedyPolicy(pomdp)


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

script_id = :solve_windfarm_greedy_nopriorbelief
plot_WindFarmPOMDP_policy!(script_id, wfparams, actions_history, rewards_history, b0,)

# @time _, info = action_info(planner, b0, tree_in_info=true)
# @time _, info = action_info(planner, b0, tree_in_info=true)
# @profview _, info = action_info(planner, b0, tree_in_info=true)

# @code_warntype _, info = action_info(planner, b0, tree_in_info=true)
# tree = info[:tree]
# inbrowser(D3Tree(tree, init_expand=1), "firefox")
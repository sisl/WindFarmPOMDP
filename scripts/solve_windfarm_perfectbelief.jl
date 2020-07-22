using POMDPs, POMDPModelTools, POMDPSimulators, POMDPPolicies, POMDPModelTools
using BasicPOMCP, ARDESPOT, POMCPOW
# using D3Trees, ProfileView

include("../src/windfarmpomdp.jl")
include("../src/beliefstates.jl")
include("../src/windfarm_expertpolicies.jl")

# Construct POMDP
no_of_sensors = 5
delta = 220
wfparams = WindFarmBeliefInitializerParams(nx=20,ny=20, grid_dist_obs = 220)
pomdp = WindFarmPOMDP(wfparams.nx, wfparams.ny, wfparams.grid_dist, wfparams.altitudes, no_of_sensors, delta)

# Get initial belief distribution (sparse version of GWA data) and initial state
b0 = initialize_belief_sparse(wfparams)
s0 = initialize_state(wfparams)

# Construct Belief Updater
up = WindFarmBeliefUpdater(wfparams.grid_dist)

# Define Solver
rollout_policy = WindFarmRolloutPolicy(pomdp)
tree_queries = 50
# solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(RandomSolver()), 0.0, check_terminal=true, consistency_fix_thresh=0.1))
# solver = POMCPSolver(tree_queries=tree_queries)
solver = POMCPOWSolver(tree_queries=tree_queries,
                       check_repeat_obs=true, 
                       check_repeat_act=true, 
                       k_action=2.0, 
                       alpha_action=0.5,
                       estimate_value=POMCPOW.RolloutEstimator(rollout_policy))


planner = solve(solver, pomdp)



println("### Starting Stepthrough ###")
global actions_history = []
global obs_history = []
global rewards_history = []
for (s, a, r, o, b, t) in stepthrough(pomdp, planner, up, b0, s0, "s,a,r,o,b,t", max_steps=no_of_sensors)
    # @show s
    @show a
    @show o
    @show r
    @show t
    push!(actions_history, a)
    push!(obs_history, o)
    push!(rewards_history, r)
end

plot_WindFarmPOMDP_policy!(wfparams, actions_history, rewards_history, b0)

# @time _, info = action_info(planner, b0, tree_in_info=true)
# @time _, info = action_info(planner, b0, tree_in_info=true)
# @profview _, info = action_info(planner, b0, tree_in_info=true)

# @code_warntype _, info = action_info(planner, b0, tree_in_info=true)
# tree = info[:tree]
# inbrowser(D3Tree(tree, init_expand=1), "firefox")

using POMDPs, POMDPModelTools, POMDPSimulators, POMDPPolicies
using BasicPOMCP, ARDESPOT, POMCPOW
using D3Trees, ProfileView

include("../src/windfarmpomdp.jl")
include("../src/beliefstates.jl")

# Construct POMDP
timesteps = 5
delta = 220
wfparams = WindFarmBeliefInitializerParams(nx=90,ny=90)
pomdp = WindFarmPOMDP(wfparams.nx, wfparams.ny, wfparams.grid_dist, wfparams.altitudes, timesteps, delta)

# Get initial belief distribution (sparse version of GWA data) and initial state
b0 = initialize_belief(wfparams)
s0 = initialize_state(wfparams)

# Construct Belief Updater
up = WindFarmBeliefUpdater(wfparams.grid_dist)

# Define Solver
# solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(RandomSolver()), 0.0, check_terminal=true, consistency_fix_thresh=0.1))
solver = POMCPSolver(tree_queries=15)
# solver = POMCPOWSolver(tree_queries=15, check_repeat_obs=true, check_repeat_act=true, k_action=3.0, alpha_action=0.2)


planner = solve(solver, pomdp)




println("### Starting Stepthrough ###")
global sg
global bg
global ag
global sgg
for (s, a, r, b, t) in stepthrough(pomdp, planner, up, b0, s0, "s,a,r,b,t", max_steps=timesteps)
    # @show s
    @show a
    @show r
    @show t
    global sg = s
    global bg = b
    global ag = a
end

# @time _, info = action_info(planner, b0, tree_in_info=true)
# @time _, info = action_info(planner, b0, tree_in_info=true)
# @profview _, info = action_info(planner, b0, tree_in_info=true)
# # @code_warntype _, info = action_info(planner, b0, tree_in_info=true)
# tree = info[:tree]
# inbrowser(D3Tree(tree, init_expand=1), "firefox")
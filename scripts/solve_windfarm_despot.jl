using POMDPs, POMDPModelTools, POMDPSimulators, POMDPPolicies
using BasicPOMCP, ARDESPOT

include("../src/windfarmpomdp.jl")
include("../src/beliefstates.jl")

# Construct POMDP
timesteps = 5
delta = 220
wfparams = WindFarmBeliefInitializerParams()
Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
pomdp = WindFarmPOMDP(wfparams.nx, wfparams.ny, wfparams.grid_dist, wfparams.altitudes, timesteps, delta, Map)

# Get initial belief distribution (sparse version of GWA data) and initial state
b0 = initialize_belief(wfparams)
s0 = initialize_state(b0)

# Construct Belief Updater
up = WindFarmBeliefUpdater(Map, wfparams.grid_dist)

# Define Solver
# solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(RandomSolver()), 0.0, check_terminal=true, consistency_fix_thresh=0.1))
solver = POMCPSolver(tree_queries=15)

planner = solve(solver, pomdp)



println("### Starting Stepthrough ###")

for (s, a, r, b, t) in stepthrough(pomdp, planner, up, b0, s0, "s,a,r,b,t", max_steps=timesteps)
    @show s
    @show a
    @show r
    @show t
end

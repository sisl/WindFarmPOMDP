using POMDPs, POMDPModelTools, POMDPSimulators, POMDPPolicies
using BasicPOMCP, POMCPOW
# using D3Trees, ProfileView

include("../src/windfarmpomdp.jl")
include("../src/beliefstates.jl")
include("../src/windfarm_expertpolicies.jl")

# Construct POMDP
no_of_sensors = 5
delta = 220 * 4
wfparams = WindFarmBeliefInitializerParams(nx=20,ny=20, grid_dist_obs = 220)
pomdp = WindFarmPOMDP(wfparams.nx, wfparams.ny, wfparams.grid_dist, wfparams.altitudes, no_of_sensors, delta)

# Get initial belief distribution (sparse version of GWA data) and initial state
b0 = initialize_belief_sparse(wfparams)
s0 = initialize_state(b0, wfparams)

# Construct Belief Updater
up = WindFarmBeliefUpdater(wfparams.altitudes, wfparams.grid_dist)

# Define Solver
rollout_policy = WindFarmRolloutPolicy(pomdp)
tree_queries = tree_queries_generic
# solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(RandomSolver()), 0.0, check_terminal=true, consistency_fix_thresh=0.1))
# solver = POMCPSolver(tree_queries=tree_queries)
solver = POMCPOWSolver(tree_queries=tree_queries,
                       check_repeat_obs=true, 
                       check_repeat_act=true, 
                       k_action=3.0,
                       alpha_action=0.3,
                       k_observation=3.0, 
                       alpha_observation=0.3,
                       criterion = MaxUCB(1.0),
                       estimate_value=POMCPOW.RolloutEstimator(rollout_policy))


planner = solve(solver, pomdp)

function BasicPOMCP.extract_belief(bu::WindFarmRolloutUpdater, node::BeliefNode)
    # global GNode = node
    s = rand(node.tree.sr_beliefs[2].dist)[1]                                       # rand simply extracts here. it is deterministic.
    return initialize_belief_rollout(s)
end

println("### Starting Stepthrough ###")
global states_history = []
global actions_history = []
global obs_history = []
global rewards_history = []
global belief_history = []
for (s, a, r, o, b, t, sp, bp) in stepthrough(pomdp, planner, up, b0, s0, "s,a,r,o,b,t,sp,bp", max_steps=no_of_sensors)
    # @show s
    @show a
    @show o
    @show r
    @show t
    push!(states_history, sp)
    push!(actions_history, a)
    push!(obs_history, o)
    push!(rewards_history, r)
    push!(belief_history, bp)
end

script_id = :solve_windfarm_pomcpow_perfectbelief
# plot_WindFarmPOMDP_policy!(script_id, wfparams, actions_history, rewards_history, b0)

# @time _, info = action_info(planner, b0, tree_in_info=true)
# @time _, info = action_info(planner, b0, tree_in_info=true)
# @profview _, info = action_info(planner, b0, tree_in_info=true)

# @code_warntype _, info = action_info(planner, b0, tree_in_info=true)
# tree = info[:tree]
# inbrowser(D3Tree(tree, init_expand=1), "firefox")

# a = rand(POMDPs.actions(pomdp))
# @profview POMDPs.gen(DDNOut(:sp, :o, :r), pomdp, s0, a, Random.GLOBAL_RNG)
# @profview POMDPs.gen(DDNOut(:sp, :o, :r), pomdp, s0, a, Random.GLOBAL_RNG)
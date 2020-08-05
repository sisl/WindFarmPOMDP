using POMDPs, POMDPModelTools, POMDPSimulators, POMDPPolicies, POMDPModelTools
using MCTS
# using D3Trees, ProfileView

# WindFarmPOMDP scripts
include("../src/windfarmpomdp.jl")
include("../src/beliefstates.jl")
include("../src/windfarm_expertpolicies.jl")

# BayesOptMCTSExperiments scripts
include("../../BayesOptMCTSExperiments/src/BOMCP.jl")
using Main.BOMCP

# Construct POMDP
no_of_sensors = 5
delta = 220 * 4
wfparams = WindFarmBeliefInitializerParams(nx=20,ny=20)
pomdp = WindFarmPOMDP(wfparams.nx, wfparams.ny, wfparams.grid_dist, wfparams.altitudes, no_of_sensors, delta)

# Get initial belief distribution (sparse version of GWA data) and initial state
b0 = initialize_belief_sparse(wfparams)
s0 = initialize_state(b0, wfparams)

# Construct Belief Updater
up = WindFarmBeliefUpdater(wfparams.altitudes, wfparams.grid_dist)

# Define Solver
rollout_policy = WindFarmRolloutPolicy(pomdp)
tree_queries = tree_queries_generic


function BOMCP.belief_type(::WindFarmBeliefUpdater)
    return WindFarmBelief
end

function BOMCP.vectorize!(v, dims, a::CartesianIndex{3})
    v[:] = collect(a.I)
    return v
end

# function BOMCP.vectorize!(v, dims, b::WindFarmBelief)
#     v .-= 10.0
    
#     # global vv = v
#     # global bb = b

#     X = b.gpla_wf.x
#     Y = b.gpla_wf.y
#     n = length(Y)
#     # @show size(v)
#     for i = 1:n
#         v[i*4 - 3] = X[1,i]
#         v[i*4 - 2] = X[2,i]
#         v[i*4 - 1] = X[3,i]
#         v[i*4] = Y[i]
#     end

#     return v
# end

function BOMCP.reward(m::WindFarmPOMDP, s::WindFarmState, a::CartesianIndex{3})
    rng = MersenneTwister()
    sp_o_r = POMDPs.gen(m, s, a, rng)    # TODO: Is this the correct BOMCP reward?
    return sp_o_r.r
end

# belief_dims = (size(b0.gpla_wf.x, 2) + 3 * no_of_sensors * length(wfparams.altitudes)) * 4     # total belief size would be the initial observations plus observations to be made; each occupy for entries: x,y,z coordinates, and the obs value.

action_selector = BOMCP.BOActionSelector(3, # action dims
                                60, #belief dims,
                                true, #discrete actions
                                kernel_params=[log(20.0), 0.0],
                                k_neighbors = 5,
                                belief_λ = -1.0)

solver = BOMCP.BOMCPSolver(action_selector, up,
                                depth=no_of_sensors,
                                n_iterations=tree_queries,
                                exploration_constant=1.0,
                                k_belief = 2.0,
                                alpha_belief = 0.1,
                                k_action=3.0,
                                alpha_action=0.3,
                                estimate_value=BOMCP.RolloutEstimator(rollout_policy))


planner = POMDPs.solve(solver, pomdp)

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

script_id = :solve_windfarm_bomcp_sparsebelief
# plot_WindFarmPOMDP_policy!(script_id, wfparams, actions_history, rewards_history, b0)

# @time _, info = action_info(planner, b0, tree_in_info=true)
# @time _, info = action_info(planner, b0, tree_in_info=true)
# @profview _, info = action_info(planner, b0, tree_in_info=true)

# @code_warntype _, info = action_info(planner, b0, tree_in_info=true)
# tree = info[:tree]
# inbrowser(D3Tree(tree, init_expand=1), "firefox")






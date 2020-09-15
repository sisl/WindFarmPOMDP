""" 
    Sensor Placement Phase


    User Arguments:

    ARGS[1]
        `solvermethod`      Solver method for sensor placements.
    Options
        entropy             Uses Shannon Entropy (Papadopoulou et al.) to place new sensors.    [Does not use turbine layout heuristic `layoutfinder`.]
        mutualinfo          Uses Mutual Information (Krause et al.) to place new sensors.       [Does not use turbine layout heuristic `layoutfinder`.]
        diffentro           Uses Differential Entropy (Herbrich et al.) to place new sensors.   [Does not use turbine layout heuristic `layoutfinder`.]
        bayesian            TODO.
        genetic             TODO.
        pomcpow             Uses the POMCPOW tree-search method (Sunberg et al.) to place new sensors.

    ARGS[2]
        `layoutfinder`      Layout type for heuristically determining a turbine layout.
    Options
        greedy              Selects turbine locations greedily w.r.t LCB of wind speed.         [Sequential]
        genetic             Selects turbine locations with genetic algorithm.                   [Non-sequential]
        mcmc                Selects turbine locations with Metropolis-Hastings MCMC.            [Non-sequential]

    ARGS[3]
        `actpolicy`         The action branching & rollout policy to be used, if the `solvermethod` is pomcpow.
    Options
        UCB                 Uses upper bound of 90% confidence interval over wind speed belief as the policy.
        MI                  Uses Mutual Information to as the policy.


    Example Calls:
        `julia solve_SensorPP.jl entropy greedy`
        `julia solve_SensorPP.jl pomcpow mcmc MI`

"""

if Threads.nthreads() == 1
    @warn "You are not running Julia in multiple threads. Aborted.\nRun e.g. `export JULIA_NUM_THREADS=16` in Terminal before entering Julia environment."
    exit()
end

# Parse user Arguments
if isempty(ARGS)
    solvermethod, layoutfinder = :pomcpow, :greedy
else
    solvermethod, layoutfinder = Symbol.(ARGS[1:2])
end

if length(ARGS) < 3
    actpolicy = :UCB
else
    actpolicy = Symbol(ARGS[3])
end

# Load modules and scripts
include("../../SensorPlacementPhase/src/SensorPP.jl")
include("../../TurbinePlacementPhase/src/TurbinePP.jl")

# Wind Field Belief Parameters
wfparams = WindFieldBeliefParams(nx=20,ny=20)

# Turbine Layout Heuristic Parameters
tlparams = TurbineLayoutParams(layoutfinder)

# Construct POMDP
no_of_sensors = 5
delta = 220 * 4
pomdp = WindFarmPOMDP(wfparams.nx, wfparams.ny, wfparams.grid_dist, wfparams.altitudes, no_of_sensors, delta)

# Get initial belief distribution and initial state
b0 = initialize_belief_lookup(wfparams)
s0 = initialize_state(b0, wfparams)

# Construct Belief Updater
up = WindFarmBeliefUpdater(wfparams.altitudes, wfparams.grid_dist)

# Define Solver
solver = extract_solver_method(pomdp, solvermethod, actpolicy)


println("### Starting Stepthrough ###")
global states_history = []
global belief_history = []
global actions_history = []
global obs_history = []
global rewards_history = []
for (s, a, r, o, b, t, sp, bp) in stepthrough(pomdp, solver, up, b0, s0, "s,a,r,o,b,t,sp,bp", max_steps=no_of_sensors)
    # @show s
    @show a
    @show o
    @show r
    @show t
    push!(states_history,   sp)
    push!(belief_history,   bp)
    push!(actions_history,  a)
    push!(obs_history,      o)
    push!(rewards_history,  r)
end


@show RR = get_ground_truth_profit(states_history, tlparams, wfparams)

# plot_WindFarmPOMDP_policy!(solvermethod, wfparams, actions_history, rewards_history, b0)

# using D3Trees, ProfileView
# @time _, info = action_info(planner, b0, tree_in_info=true)
# @time _, info = action_info(planner, b0, tree_in_info=true)
# @profview _, info = action_info(planner, b0, tree_in_info=true)

# @code_warntype _, info = action_info(planner, b0, tree_in_info=true)
# tree = info[:tree]
# inbrowser(D3Tree(tree, init_expand=1), "firefox")
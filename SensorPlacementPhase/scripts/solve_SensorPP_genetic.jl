""" 
    Sensor Placement Phase


    User Arguments:

    ARGS[1]
        `solvermethod`      Solver method for sensor placements.
    Options
        genetic             Uses genetic algorithm.

    ARGS[2]
        `layoutfinder`      Layout type for heuristically determining a turbine layout.
    Options
        greedy              Selects turbine locations greedily w.r.t LCB of wind speed.         [Sequential]
        genetic             Selects turbine locations with genetic algorithm.                   [Non-sequential]
        mcmc                Selects turbine locations with Metropolis-Hastings MCMC.            [Non-sequential]


    Example Calls:
        `julia solve_SensorPP.jl genetic mcmc`

"""

if Threads.nthreads() == 1
    @warn "You are not running Julia in multiple threads. Aborted.\nRun e.g. `export JULIA_NUM_THREADS=16` in Terminal before entering Julia environment."
    exit()
end

# Parse user Arguments
if isempty(ARGS)
    solvermethod, layoutfinder = :genetic, :mcmc
else
    solvermethod, layoutfinder = Symbol.(ARGS)
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

# Define Solver
solver = extract_solver_method(pomdp, solvermethod)

# Compute the optimal sensor placements
soln = get_solution(pomdp, s0, tlparams, wfparams, solver)

# Show utility of solution
@show RR = get_ground_truth_profit(s0, soln, tlparams, wfparams)



# plot_WindFarmPOMDP_policy!(solvermethod, wfparams, actions_history, rewards_history, b0)

# using D3Trees, ProfileView
# @time _, info = action_info(planner, b0, tree_in_info=true)
# @time _, info = action_info(planner, b0, tree_in_info=true)
# @profview _, info = action_info(planner, b0, tree_in_info=true)

# @code_warntype _, info = action_info(planner, b0, tree_in_info=true)
# tree = info[:tree]
# inbrowser(D3Tree(tree, init_expand=1), "firefox")
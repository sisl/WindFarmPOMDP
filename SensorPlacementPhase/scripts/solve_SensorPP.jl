""" 
    Sensor Placement Phase
"""

@show VERSION
include("../../SensorPlacementPhase/src/SensorPP.jl")
include("../../TurbinePlacementPhase/src/TurbinePP.jl")

# Parse ARGS
CMD_ARGS = parse_commandline()
@show_args CMD_ARGS

# Record time elapsed
time_taken = @elapsed begin

# Wind Field Belief Parameters
wfparams = WindFieldBeliefParams(farm=CMD_ARGS[:farm], noise_seed = CMD_ARGS[:noise_seed])

# Turbine Layout Heuristic Parameters
tlparams = TurbineLayoutParams(CMD_ARGS[:layoutfinder])

# Construct POMDP
no_of_sensors = CMD_ARGS[:sensors]
delta = 220 * 4
pomdp = WindFarmPOMDP(wfparams, no_of_sensors, delta)

# Get initial belief distribution and initial state
b0 = initialize_belief_lookup(wfparams)
s0 = initialize_state(b0, wfparams)

# Construct Belief Updater
up = WindFarmBeliefUpdater(wfparams.altitudes, wfparams.grid_dist)

# Define Solver
solver = extract_solver_method(pomdp, CMD_ARGS[:solvermethod], CMD_ARGS[:actpolicy], CMD_ARGS[:tree_queries])

# Retrieve Solution
RR, soln = retrieve_solution_from_solver(CMD_ARGS[:solvermethod], tlparams, wfparams, pomdp, solver, up, b0, s0, no_of_sensors)
end    # time_taken


# Save results
@show round(time_taken; digits=3)
writedlm_append(CMD_ARGS[:savename], hcat(CMD_ARGS[:solvermethod], CMD_ARGS[:layoutfinder], CMD_ARGS[:noise_seed], CMD_ARGS[:actpolicy], CMD_ARGS[:tree_queries], CMD_ARGS[:sensors], size(soln, 2), round(time_taken; digits=2), RR))
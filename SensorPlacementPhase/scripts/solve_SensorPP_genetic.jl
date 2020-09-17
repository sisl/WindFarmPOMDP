""" 
    Sensor Placement Phase


    User Arguments:

    ARGS[1]
        `solvermethod`      Solver method for sensor placements. Inherited automatically from solve_SensorPP.jl.
    Options
        genetic             Uses genetic algorithm to place new sensors.

    ARGS[2]
        `layoutfinder`      Layout type for heuristically determining a turbine layout.
    Options
        greedy              Selects turbine locations greedily w.r.t LCB of wind speed.         [Sequential]
        genetic             Selects turbine locations with genetic algorithm.                   [Non-sequential]
        mcmc                Selects turbine locations with Metropolis-Hastings MCMC.            [Non-sequential]

    ARGS[3]
        `no_of_iterations`  Number of times genetic algorithm is called.

    ARGS[4]
        `populationSize`    Size of the population during each call of the genetic algorithm.
    
    ARGS[5]
        `crossoverRate`     Rate of crossover, between 0 and 1.
    
    ARGS[6]
        `mutationRate`      Rate of mutation, between 0 and 1.

    ARGS[7]
        `savename`          Save name for results. Any valid String accepted. Enter 0 to skip saving.


    Example Call:
        `julia solve_SensorPP_genetic.jl genetic mcmc 3 50 0.8 0.05 foo.csv`

"""

# Parse ARGS
localARGS = isdefined(Main, :genericARGS) ? genericARGS : ARGS
@show solvermethod, layoutfinder = Symbol.(localARGS[1:2])
@show extra_params = localARGS[3:end-1]
@show savename = localARGS[end]

include("../../SensorPlacementPhase/src/SensorPP.jl")
include("../../TurbinePlacementPhase/src/TurbinePP.jl")

# Record time elapsed
time_taken = @elapsed begin

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
solver = extract_solver_method(pomdp, solvermethod, extra_params)

# Compute the optimal sensor placements
soln = get_solution(s0, pomdp, tlparams, wfparams, solver)

# Show utility of solution
@show RR = get_ground_truth_profit(s0, soln, tlparams, wfparams)
end    # time_taken


# Save results
@show round(time_taken; digits=3)
writedlm_append(savename, hcat(solvermethod, layoutfinder, vec(extra_params)..., round(time_taken; digits=2), RR))
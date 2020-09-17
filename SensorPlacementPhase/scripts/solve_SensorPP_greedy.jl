""" 
    Sensor Placement Phase


    User Arguments:

    ARGS[1]
        `solvermethod`      Solver method for sensor placements. Inherited automatically from solve_SensorPP.jl.
    Options
        entropy             Uses Shannon Entropy (Papadopoulou et al.) to place new sensors.    [Uses `layoutfinder` only after calculating ground truth profit, not during placements.]
        mutualinfo          Uses Mutual Information (Krause et al.) to place new sensors.       [Uses `layoutfinder` only after calculating ground truth profit, not during placements.]
        diffentro           Uses Differential Entropy (Herbrich et al.) to place new sensors.   [Uses `layoutfinder` only after calculating ground truth profit, not during placements.]

    ARGS[2]
        `layoutfinder`      Layout type for heuristically determining a turbine layout.
    Options
        greedy              Selects turbine locations greedily w.r.t LCB of wind speed.         [Sequential]
        genetic             Selects turbine locations with genetic algorithm.                   [Non-sequential]
        mcmc                Selects turbine locations with Metropolis-Hastings MCMC.            [Non-sequential]

    ARGS[3]
        `savename`          Save name for results. Any valid String accepted. Enter 0 to skip saving.


    Example Call:
        `julia solve_SensorPP_greedy.jl entropy greedy foo.csv`

"""

# Parse ARGS
localARGS = isdefined(Main, :genericARGS) ? genericARGS : ARGS
@show solvermethod, layoutfinder = Symbol.(localARGS[1:2])
@show savename = localARGS[3]

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

# Construct Belief Updater
up = WindFarmBeliefUpdater(wfparams.altitudes, wfparams.grid_dist)

# Define Solver
solver = extract_solver_method(pomdp, solvermethod)


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

# Show utility of solution
@show RR = get_ground_truth_profit(states_history, tlparams, wfparams)
end    # time_taken


# Save results
@show round(time_taken; digits=3)
writedlm_append(savename, hcat(solvermethod, layoutfinder, round(time_taken; digits=2), RR))
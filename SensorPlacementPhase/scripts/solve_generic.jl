"""
The user should pass in the the following arguments in the Terminal.

# Arguments 
- `no_of_runs` number of runs for the script specified
- `script_name` the name of the script to run 
- any additional arguments entering the script `script_name`
"""

if Threads.nthreads() == 1
    @warn "You are not running Julia in multiple threads. Aborted.\nRun e.g. `export JULIA_NUM_THREADS=16` in Terminal before running script."
    exit()
end

include("../../SensorPlacementPhase/src/SensorPP.jl")
include("../../TurbinePlacementPhase/src/TurbinePP.jl")

no_of_runs = parse(Int, ARGS[1])
script_name = ARGS[2]

dir = string("Generic_Run_Results")
!isdir(dir) ? mkdir(dir) : nothing
dt = Dates.now()
savename = string("./$dir/", script_name, '_', dt, ".csv")

genericARGS = vcat(ARGS[3:end]..., savename)


global REWARD_RECORDS = []
for c in 1:no_of_runs
    @show c
    include(script_name)
    # run(`julia $(script_name) $(additional_args) $(savename)`)
end
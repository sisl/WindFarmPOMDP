""" 
    Creates the plots used in the journal publication.
"""

@show VERSION
include("../../SensorPlacementPhase/src/SensorPP.jl")
include("../../TurbinePlacementPhase/src/TurbinePP.jl")

using Plots

sensor_color = :white
turbine_color = :lightgreen
savetype = :png

!isdir("Figures") ? mkdir("Figures") : nothing
# dir = string("Figures/", Dates.now())
# mkdir(dir)
dir = string("Figures")


# Parse ARGS
CMD_ARGS = parse_commandline()
@show_args CMD_ARGS

# Params
farm_name = "AltamontCA"
wfparams = WindFieldBeliefParams(farm=farm_name, noise_seed=CMD_ARGS[:noise_seed])
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


######### Plots of AltamontCA #########
h = 150
HEATMAP_CLIM_AltamontCA = (4.0, 11.0)

# Plot1: Entire AltamontCA
Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
X_field, Y_field = get_dataset(Map, [h], wfparams.grid_dist, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)
p1 = heatmap(Map[h], clim=HEATMAP_CLIM_AltamontCA, size=(1800,1200), xaxis=false, yaxis=false, legend = :none) #, title="Altamont Pass, CA, h = $(h)m") 
Plots.savefig(p1, "./$dir/p1_h$(h).$savetype")

# Plot2: Cropped nx × ny AltamontCA
b = initialize_belief_perfect(wfparams)
gpla_wf = b.gpla_wf
μ, σ² = GaussianProcesses.predict_f(gpla_wf, X_field)
σ = sqrt.(σ²)
p2 = Plots.heatmap(reshape(μ, (wfparams.nx, wfparams.ny)), clim=HEATMAP_CLIM_AltamontCA, size=(1800,1200), xaxis=false, yaxis=false, legend = :none) #, title="Altamont Pass, CA, h = $(h)m, lower left 20×20") 
Plots.savefig(p2, "./$dir/p2_h$(h).$savetype")

######################################


######### Plots of initial belief #########

# Plot3: Initial belief, no noise
b = initialize_belief_lookup(wfparams; has_noise=false)
gpla_wf = b.gpla_wf
μ, σ² = GaussianProcesses.predict_f(gpla_wf, X_field)
σ = sqrt.(σ²)
p3 = Plots.heatmap(reshape(μ, (wfparams.nx, wfparams.ny)), clim=HEATMAP_CLIM_AltamontCA, size=(1800,1200), xaxis=false, yaxis=false, legend = :none) #, title="Altamont Pass, CA, h = $(h)m, lower left 20×20, low-resolution") 
Plots.savefig(p3, "./$dir/p3_h$(h).$savetype")

# Plot4: Initial belief, with noise
b = initialize_belief_lookup(wfparams; has_noise=true)
gpla_wf = b.gpla_wf
μ, σ² = GaussianProcesses.predict_f(gpla_wf, X_field)
σ = sqrt.(σ²)
p4 = Plots.heatmap(reshape(μ, (wfparams.nx, wfparams.ny)), clim=HEATMAP_μ_CLIM, size=(1800,1200), xaxis=false, yaxis=false, legend = :none) #, title="Altamont Pass, CA, h = $(h)m, lower left 20×20, low-resolution, with noise") 
Plots.savefig(p4, "./$dir/p4_h$(h).$savetype")

###########################################


######### Plot of Power Curve Approx. #########

# Plot5: Power Curve Approx
poly_degree = 3
data_dir = "../../TurbinePlacementPhase/data/ge_turbine_power_curve.csv"
power_data = DelimitedFiles.readdlm(data_dir, Float64, comments=true, comment_char='#')
speed_ms, power_kW = eachcol(power_data)
curve_fit = Polynomials.fit(speed_ms, power_kW, poly_degree)
p5 = scatter(speed_ms, power_kW, label="Data")
plot!(curve_fit, extrema(speed_ms)..., label="Fit", legend=:bottomright)
xlims!(2,12)
title!("Power Curve Approximation, GE 2.5-120 Turbine")
xlabel!("Wind Speed [m/s]")
ylabel!("Power Output [kW]")
Plots.savefig(p5, "./$dir/p5.$savetype")

############################################


######### LCB to Power relation #########

# Plot6: Sample plot showing LCB
xs = [1, 2, 3]
μs = [20, 15, 17]
σs = [13, 2, 5]
lb = μs - σs
ub = μs + σs
p6 = plot(xs,μs,grid=false, ribbon=σs, ribboncolor=:red, fillalpha=.5, ylims=(minimum(lb)-1,maximum(ub)+1), yaxis=nothing, xaxis=nothing, label="Mean Function")
scatter!(xs, ub, color=:green, label="Upper Bound")
scatter!(xs, lb, color=:red, label="Lower Bound")
plot!(xs, μs - (μs-lb)*0.9, color=:orange, label="90% LCB")
xlabel!("Location")
ylabel!("Wind Speed Mean and Variance in Belief")
title!("Example Belief of Wind Speed, with Bounds")
Plots.savefig(p6, "./$dir/p6.$savetype")

############################################


######### Belief Plots #########

# Plot7: Entropy

a1 = CartesianIndex(1980, 1980, 200)
a2 = CartesianIndex(4180, 4180, 200)
a3 = CartesianIndex(0, 4180, 200)
a4 = CartesianIndex(0, 0, 200)
a5 = CartesianIndex(4180, 0, 200)

b1 = POMDPs.update(up,b0,a1,[0,0,0])
b2 = POMDPs.update(up,b1,a2,[0,0,0])
b3 = POMDPs.update(up,b2,a3,[0,0,0])
b4 = POMDPs.update(up,b3,a4,[0,0,0])
b5 = POMDPs.update(up,b4,a5,[0,0,0])

a_history = [a1, a2, a3, a4, a5]
b_history = [b1, b2, b3, b4, b5]

# These are saved to /SensorPlacementPhase/scripts/Figures/
plot_WindFarmPOMDP_belief_history_pub(wfparams, a_history, b_history, b0)


# Plot8: Mutual Info

a1 = CartesianIndex(1980, 1980, 200)
a2 = CartesianIndex(3960, 3960, 200)
a3 = CartesianIndex(220, 3960, 200)
a4 = CartesianIndex(220, 220, 200)
a5 = CartesianIndex(3960, 220, 200)

b1 = POMDPs.update(up,b0,a1,[0,0,0])
b2 = POMDPs.update(up,b1,a2,[0,0,0])
b3 = POMDPs.update(up,b2,a3,[0,0,0])
b4 = POMDPs.update(up,b3,a4,[0,0,0])
b5 = POMDPs.update(up,b4,a5,[0,0,0])

a_history = [a1, a2, a3, a4, a5]
b_history = [b1, b2, b3, b4, b5]

# These are saved to /SensorPlacementPhase/scripts/Figures/
plot_WindFarmPOMDP_belief_history_pub(wfparams, a_history, b_history, b0)


# Plot9: Diffentro

a1 = CartesianIndex(1980, 1980, 200)
a2 = CartesianIndex(3080, 3080, 200)
a3 = CartesianIndex(880, 3080, 200)
a4 = CartesianIndex(880, 880, 200)
a5 = CartesianIndex(3080, 880, 200)

b1 = POMDPs.update(up,b0,a1,[0,0,0])
b2 = POMDPs.update(up,b1,a2,[0,0,0])
b3 = POMDPs.update(up,b2,a3,[0,0,0])
b4 = POMDPs.update(up,b3,a4,[0,0,0])
b5 = POMDPs.update(up,b4,a5,[0,0,0])

a_history = [a1, a2, a3, a4, a5]
b_history = [b1, b2, b3, b4, b5]

# These are saved to /SensorPlacementPhase/scripts/Figures/
plot_WindFarmPOMDP_belief_history_pub(wfparams, a_history, b_history, b0)

############################################

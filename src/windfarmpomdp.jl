# Modules
using POMDPs
using GaussianProcesses
using Random
using Distributions
using Discretizers
using Parameters
using ProgressBars
using DelimitedFiles
using Dates
using Plots

# windGP scripts
include("../../windGP/src/dataparser_GWA.jl")
include("../../windGP/src/GPLA.jl")
include("../../windGP/src/utils/WLK_SEIso.jl")

# WindFarmPOMDP scripts
include("../src/beliefstates.jl")
include("../src/utils/misc.jl")


struct WindFarmPOMDP <: POMDP{WindFarmState, CartesianIndex{3}, AbstractVector}
    nx::Int
    ny::Int
    grid_dist::Int
    altitudes::AbstractVector
    timesteps::Int
    delta::Int          # Minimum distance between actions
end

POMDPs.discount(::WindFarmPOMDP) = 0.9
POMDPs.isterminal(p::WindFarmPOMDP, s::WindFarmState) = size(s.x_acts, 2) > p.timesteps

function turbine_layout_heuristic(p::WindFarmPOMDP, s::WindFarmState, gpla_wf::GPLA)
    λ = 1.0    # TODO: Coefficient for estimating profit.
    no_of_turbines = 10   # TODO: Change this?
    
    X_field = CartIndices_to_Array(POMDPs.actions(p, s))
    μ, σ² = GaussianProcesses.predict_f(gpla_wf, X_field)
    σ = sqrt.(σ²)
    N = length(gpla_wf.y)

    z_value = 1.645   # chosen: 90 percent confidence interval
    
    if !isempty(s.y_obs_full)
        Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
        Y_field = get_Y_from_farm_location.(eachcol(X_field), Ref(Map), Ref(wfparams.grid_dist))
        truth_CB = μ - z_value * abs.(μ - Y_field)     # penalty for incorrect layout heuristic
    else
        truth_CB = μ
    end
    
    LCB = μ - z_value / sqrt(N) * σ
    best_vals = partialsortperm(vec(LCB), 1:no_of_turbines, rev=true)
    expected_profit = λ * sum(truth_CB[best_vals] .^3)


    # @show length(gpla_wf.y)
    # @show μ[best_vals]
    # @show σ[best_vals]
    # @show expected_profit
    return expected_profit
end

function get_tower_cost(a)
    height = a[end]
    # height < 90 ? 92600.0/116000.0 : 1.0 
    return height * 10
end

function get_GPLA_for_gen(X, Y, wfparams::WindFarmBeliefInitializerParams)
    kernel = WLK_SEIso(eps(), eps(), eps(), eps(), eps(), eps())
    gpla_wf = GPLA(X, Y, wfparams.num_neighbors, 0, 0, MeanConst(wfparams.theta[2]), kernel, wfparams.theta[1])
    GaussianProcesses.set_params!(gpla_wf, wfparams.theta)
    return gpla_wf
end


# State: WindFarmState
# Action: CartesianIndex{3}
# Obs: Int
function POMDPs.gen(m::WindFarmPOMDP, s::WindFarmState, a0::CartesianIndex{3}, rng::AbstractRNG)

    # Transform the action location to Vector
    a = expand_action_to_altitudes(a0, m.altitudes)
    a = CartIndices_to_Array(a)
    
    # Get observations
    if !isempty(s.y_obs_full)
        gpla_wf_full = get_GPLA_for_gen(s.x_obs_full, s.y_obs_full, wfparams)
        gpla_wf = get_GPLA_for_gen(s.x_obs, s.y_obs, wfparams)
        o = rand(gpla_wf_full, a)
        @show "observed truth"
    else
        gpla_wf = get_GPLA_for_gen(s.x_obs, s.y_obs, wfparams)
        o = rand(gpla_wf, a)
    end

    
    # Seperate the action's altitude and the observation at that altitude.
    a0_idx = findfirst(x -> x == a0[3], m.altitudes)
    a0 = a[:, a0_idx:a0_idx]
    o0 = o[a0_idx]
    
    # Get next state
    if isempty(s.x_acts)
        sp_x_acts = a0
    else
        sp_x_acts = hcat(s.x_acts, a0)
    end
    sp_x_obs = hcat(s.x_obs, a)
    sp_y_obs = vcat(s.y_obs, o)
    sp = WindFarmState(sp_x_acts, sp_x_obs, sp_y_obs, s.x_obs_full, s.y_obs_full)

    # Discretize observation to avoid shallow trees
    # o = encode(LinearDiscretizer(collect(0 : 0.25 : 10)), o)  ## TODO: Use discretized FLoat instead
    o = round.(o * 2)/2   # rounds to nearest 0.5
    
    # Get reward
    GaussianProcesses.fit!(gpla_wf, sp_x_obs, sp_y_obs) 
    r = turbine_layout_heuristic(m, sp, gpla_wf) - get_tower_cost(a0)

    # @show o
    # @show a
    # @show a, o, size(s.x_acts, 2)
    return (sp = sp, o = o, r = r/10000)
end

# P(o|s,a,s')
function POMDPModelTools.obs_weight(p::WindFarmPOMDP, s::WindFarmState, a::CartesianIndex{3}, sp::WindFarmState, o::AbstractVector)
    a = CartIndices_to_Vector(a)

    gpla_wf = get_GPLA_for_gen(s.x_obs, s.y_obs, wfparams)
    μ, σ² = GaussianProcesses.predict_f(gpla_wf, a)
    σ = sqrt.(σ²)
    
    return Distributions.pdf(Normal(μ..., σ...), average(o))
end


############################
## Possible Action Spaces ##
############################

function POMDPs.actions(p::WindFarmPOMDP)
    """ All possible actions, regardless of history. """
    permitted_altitudes = p.altitudes       # TODO: Change this?
    permitted_xy_dists = p.grid_dist        # TODO: Change this?

    x_ranges = 0 : permitted_xy_dists : p.grid_dist * p.nx-1
    y_ranges = 0 : permitted_xy_dists : p.grid_dist * p.ny-1
    all_actions = [CartesianIndex(x,y,z) for x in x_ranges for y in y_ranges for z in permitted_altitudes]

    return vec(collect(all_actions))
end

function POMDPs.actions(p::WindFarmPOMDP, s::WindFarmState)
    """ Permitted actions, after having taken previous hallucination actions in the tree. """

    all_actions_Set = Set(POMDPs.actions(p))

    x_acts_cartidx = Vector_to_CartIndices.(eachcol(s.x_acts))
    expanded = expand_action_to_limits.(x_acts_cartidx, Ref(p.altitudes), Ref(p.grid_dist), Ref(p.delta))

    setdiff!(all_actions_Set, Set(vcat(expanded...)))
    return collect(all_actions_Set)
end

function POMDPs.actions(p::WindFarmPOMDP, b::WindFarmBelief)
    """ Permitted actions, after having taken an actual action, thereby updating the belief. """

    all_actions_Set = Set(POMDPs.actions(p))
    
    x_acts_cartidx = Vector_to_CartIndices.(eachcol(b.x_acts))
    expanded = expand_action_to_limits.(x_acts_cartidx, Ref(p.altitudes), Ref(p.grid_dist), Ref(p.delta))

    setdiff!(all_actions_Set, Set(vcat(expanded...)))
    return collect(all_actions_Set)
end

expand_action_to_altitudes(a::CartesianIndex, altitudes::AbstractVector) = [CartesianIndex(a[1], a[2], h) for h in altitudes]
expand_action_to_other_altitudes(a::CartesianIndex, altitudes::AbstractVector) = [CartesianIndex(a[1], a[2], h) for h in setdiff(Set(altitudes),a[3])]

function expand_action_to_limits(a::CartesianIndex, altitudes, grid_dist, delta)
    """ Creates an array of blocked locations (considering both altitude and delta limits), given an action. """
    a1, a2 = a[1], a[2]
    block = [CartesianIndex(a1+δ1, a2+δ2, h) for h in altitudes for δ1 in -delta:grid_dist:delta for δ2 in -delta:grid_dist:delta]
    return block
end

function plot_WindFarmPOMDP_policy!(script_id::Symbol, wfparams::WindFarmBeliefInitializerParams, actions_history::AbstractArray, rewards_history::AbstractArray, b0::WindFarmBelief)
    println("### Creating Policy Plots ###")
    nx, ny = wfparams.nx, wfparams.ny
    Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
    actions_history = CartIndices_to_Vector.(actions_history)
    
    !isdir("Figures") ? mkdir("Figures") : nothing
    dir = string("Figures/", Dates.now())
    mkdir(dir)

    gpla_wf = b0.gpla_wf
    
    for h in wfparams.altitudes

        a_in_h = reshape(Float64[],2,0)
        for a in actions_history
            if a[end]==h
                a_in_h = hcat(a_in_h, a[1:2] / wfparams.grid_dist + [1,1])    # [1,1] is added because locations were 0-based indexed.
            end
        end
        
        X_field, Y_field = get_dataset(Map, [h], wfparams.grid_dist, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)
        p = Plots.heatmap(reshape(Y_field, (nx,ny)), title="Wind Farm Sensor Locations Chosen, h = $(h)m")
        Plots.scatter!(a_in_h[2,:], a_in_h[1,:], legend=false, color=:white)  # Notice that the row and col of `a_in_h` is reversed.
        Plots.savefig(p, "./$dir/Plot_$h")
    
        # Plots of initial belief below.
        μ, σ² = GaussianProcesses.predict_f(gpla_wf, X_field)
        σ = sqrt.(σ²)

        p2 = Plots.heatmap(reshape(σ, (nx,ny)), title="Wind Farm Initial Belief Variance, h = $(h)m")
        Plots.scatter!(a_in_h[2,:], a_in_h[1,:], legend=false, color=:white)  # Notice that the row and col of `a_in_h` is reversed.
        Plots.savefig(p2, "./$dir/Plot2_$h")

        p3 = Plots.heatmap(reshape(μ, (nx,ny)), title="Wind Farm Initial Belief Mean, h = $(h)m")
        Plots.scatter!(a_in_h[2,:], a_in_h[1,:], legend=false, color=:white)  # Notice that the row and col of `a_in_h` is reversed.
        Plots.savefig(p3, "./$dir/Plot3_$h")

    end
    println("### Policy Plots Saved to $dir ###")

    save_rewards_to_disk(script_id, rewards_history, "./$dir/rewards.txt")
    return nothing
end

function save_rewards_to_disk(script_id::Symbol, rewards_history::AbstractArray, savename::String)
    println("### Total Rewards: $(sum(rewards_history)) ###")
    open(savename, "a") do io
        writedlm(io, ["Script: ", string(script_id, ".jl"), ""])
        writedlm(io, ["History: ", rewards_history..., ""])
        writedlm(io, ["Total: ", sum(rewards_history)])
    end
    return nothing
end
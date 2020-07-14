# Modules
using POMDPs
using GaussianProcesses
using Random
using Distributions
using Discretizers
using Parameters
using ProgressBars

# windGP scripts
include("../../windGP/src/dataparser_GWA.jl")
include("../../windGP/src/GPLA.jl")
include("../../windGP/src/utils/WLK_SEIso.jl")

# WindFarmPOMDP scripts
include("../src/beliefstates.jl")
include("../src/utils/misc.jl")


struct WindFarmPOMDP <: POMDP{WindFarmState, CartesianIndex{3}, Number}
    nx::Int
    ny::Int
    grid_dist::Int
    altitudes::AbstractVector
    timesteps::Int
    delta::Int          # Minimum distance between actions
end

POMDPs.discount(::WindFarmPOMDP) = 0.9
POMDPs.isterminal(p::WindFarmPOMDP, s::WindFarmState) = size(s.x_acts, 2) > p.timesteps

function expertPolicy(gpla_wf::GPLA)
    λ = 1.0    # TODO: Coefficient for estimating profit.
    no_of_turbines = 10   # TODO: Change this?
    ground_truth = s0

    # X_field, Y_field = get_dataset(Map, m.altitudes, m.grid_dist, m.grid_dist, 1, m.nx, 1, m.ny)
    X_field = s0.x_obs

    μ, σ² = GaussianProcesses.predict_f(gpla_wf, X_field)
    σ = sqrt.(σ²)
    N = length(σ)

    z_value = 1.645   # chosen: 90 percent confidence interval
    LCB = μ - z_value / sqrt(N) * σ
    
    best_vals = partialsortperm(vec(LCB), 1:no_of_turbines, rev=true)
    expected_profit = λ * sum(μ[best_vals] .^3)
    
    # @show sort(vec(μ), rev=true)[1:10]
    # @show μ[best_vals]
    # @show σ[best_vals]
    @show expected_profit
    return expected_profit
end

function get_tower_cost(a)
    height = a[end]
    height < 90 ? 92600.0/116000.0 : 1.0 
end

function get_GPLA_for_gen(s::WindFarmState, wfparams::WindFarmBeliefInitializerParams)
    kernel = WLK_SEIso(eps(), eps(), eps(), eps(), eps(), eps())
    gpla_wf = GPLA(s.x_obs, s.y_obs, wfparams.num_neighbors, 0, 0, MeanConst(wfparams.theta[2]), kernel, wfparams.theta[1])
    GaussianProcesses.set_params!(gpla_wf, wfparams.theta)
    return gpla_wf
end


# State: WindFarmState
# Action: CartesianIndex{3}
# Obs: Int
function POMDPs.gen(m::WindFarmPOMDP, s::WindFarmState, a::CartesianIndex{3}, rng::AbstractRNG)

    # Transform the action location to Vector
    a = CartIndices_to_Vector(a)

    # Get observation
    gpla_wf = get_GPLA_for_gen(s, wfparams)
    # o, _ = predict_f(gpla_wf, a)
    o = rand(gpla_wf, a)
    o = o[1]
    @show o
    @show a
    
    # Get next state
    if isempty(s.x_acts)
        sp_x_acts = a
    else
        sp_x_acts = hcat(s.x_acts, a)
    end
    sp_x_obs = hcat(s.x_obs, a)
    sp_y_obs = vcat(s.y_obs, o)
    sp = WindFarmState(sp_x_acts, sp_x_obs, sp_y_obs)

    # Discretize observation to avoid shallow trees
    # o = encode(LinearDiscretizer(collect(0 : 0.25 : 10)), o)  ## TODO: Use discretized FLoat instead
    o = round(o * 2)/2   # rounds to nearest 0.5
    
    # Get reward
    GaussianProcesses.fit!(gpla_wf, sp_x_obs, sp_y_obs) 
    r = expertPolicy(gpla_wf) - get_tower_cost(a)

    # @show a, o, size(s.x_acts, 2)
    return (sp = sp, o = o, r = r)
end




############################
## Possible Action Spaces ##
############################

function POMDPs.actions(p::WindFarmPOMDP)
    "All possible actions, regardless of history."
    permitted_altitudes = p.altitudes       # TODO: Change this?
    permitted_xy_dists = p.delta            # TODO: Change this?

    x_ranges = 0 : permitted_xy_dists : p.grid_dist * p.nx-1
    y_ranges = 0 : permitted_xy_dists : p.grid_dist * p.ny-1
    all_actions = [CartesianIndex(x,y,z) for x in x_ranges for y in y_ranges for z in permitted_altitudes]

    return vec(collect(all_actions))
end

function POMDPs.actions(p::WindFarmPOMDP, s::WindFarmState)
    "Permitted actions, after having taken previous hallucination actions in the tree."

    all_actions_Set = Set(POMDPs.actions(p))
    x_acts_Set = Set(s.x_acts)
    setdiff!(all_actions_Set, x_acts_Set)

    return collect(all_actions_Set)
end

function POMDPs.actions(p::WindFarmPOMDP, b::WindFarmBelief)
    "Permitted actions, after having taken an actual action, thereby updating the belief."

    all_actions_Set = Set(POMDPs.actions(p))
    x_acts_Set = Set(b.x_acts)
    setdiff!(all_actions_Set, x_acts_Set)

    return collect(all_actions_Set)
end
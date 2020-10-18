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

@memoize function get_GPLA_for_gen(X, Y, wfparams::WindFieldBeliefParams)

    if typeof(b0.gpla_wf.mean) == GaussianProcesses.MeanConst
        gp_mean = MeanConst(wfparams.theta[2])
        kernel = WLK_SEIso(eps(), eps(), eps(), eps(), eps(), eps())
        gpla_wf = GPLA(X, Y, wfparams.num_neighbors, 0, 0, gp_mean, kernel, wfparams.theta[1])
        GaussianProcesses.set_params!(gpla_wf, wfparams.theta)
    else
        gp_mean = b0.gpla_wf.mean
        kernel = WLK_SEIso(eps(), eps(), eps(), eps(), eps(), eps())
        gpla_wf = GPLA(X, Y, wfparams.num_neighbors, 0, 0, gp_mean, kernel, wfparams.theta[1])
        GaussianProcesses.set_params!(gpla_wf, wfparams.theta[3:end]; noise=false, domean=false)
    end

    return gpla_wf
end

""" State: WindFarmState
    Action: CartesianIndex{3}
    Obs: Int
"""
function POMDPs.gen(m::WindFarmPOMDP, s::WindFarmState, a0::CartesianIndex{3}, rng::AbstractRNG)

    # Transform the action location to Vector
    a = expand_action_to_below_altitudes(a0, m.altitudes)
    a = CartIndices_to_Array(a)
    
    # Get observations
    if !isempty(s.y_obs_full)
        gpla_wf_full = get_GPLA_for_gen(s.x_obs_full, s.y_obs_full, wfparams)
        gpla_wf = get_GPLA_for_gen(s.x_obs, s.y_obs, wfparams)
        o = rand(gpla_wf_full, a)
        # gpla_wf = get_GPLA_for_gen(s.x_obs_full, s.y_obs_full, wfparams)
        # o = rand(gpla_wf, a)
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
    o = round.(o * 2)/2   # rounds to nearest 0.5
    
    # Get reward
    GaussianProcesses.fit!(gpla_wf, sp_x_obs, sp_y_obs)
    r = get_layout_profit(sp, gpla_wf, tlparams, wfparams)    /2.3e7/tlparams.no_of_turbines

    if r > 1 @warn "Reward surpasses 1.0, and is $r. Increase normalizing value." end
    return (sp = sp, o = o, r = r)
end

# P(o|s,a,s')
function POMDPModelTools.obs_weight(p::WindFarmPOMDP, s::WindFarmState, a::CartesianIndex{3}, sp::WindFarmState, o::AbstractVector)
    a = CartIndices_to_Vector(a)

    gpla_wf = get_GPLA_for_gen(s.x_obs, s.y_obs, wfparams)
    μ, σ² = GaussianProcesses.predict_f(gpla_wf, a)
    σ = sqrt.(σ²)
    
    return Distributions.pdf(Normal(μ..., σ...), mean(o))
end


############################
## Possible Action Spaces ##
############################

@memoize function POMDPs.actions(p::WindFarmPOMDP)
    """ All possible actions, regardless of history. """
    altitudes = p.altitudes
    grid_dist = p.grid_dist

    x_ranges = 0 : grid_dist : grid_dist * p.nx-1
    y_ranges = 0 : grid_dist : grid_dist * p.ny-1
    all_actions = [CartesianIndex(x,y,z) for x in x_ranges for y in y_ranges for z in altitudes]

    return vec(collect(all_actions))
end

@memoize function POMDPs.actions(p::WindFarmPOMDP, s::WindFarmState)
    """ Permitted actions, after having taken previous hallucination actions in the tree. """

    all_actions_Set = Set(POMDPs.actions(p))

    x_acts_cartidx = Vector_to_CartIndices.(eachcol(s.x_acts))
    expanded = expand_action_to_limits.(x_acts_cartidx, Ref(p.altitudes), Ref(p.grid_dist), Ref(p.delta))

    setdiff!(all_actions_Set, Set(vcat(expanded...)))
    return collect(all_actions_Set)
end

@memoize function POMDPs.actions(p::WindFarmPOMDP, b::WindFarmBelief)
    """ Permitted actions, after having taken an actual action, thereby updating the belief. """

    all_actions_Set = Set(POMDPs.actions(p))
    
    x_acts_cartidx = Vector_to_CartIndices.(eachcol(b.x_acts))
    expanded = expand_action_to_limits.(x_acts_cartidx, Ref(p.altitudes), Ref(p.grid_dist), Ref(p.delta))

    setdiff!(all_actions_Set, Set(vcat(expanded...)))
    return collect(all_actions_Set)
end


####################
## Expand Actions ##
####################

expand_action_to_altitudes(a::CartesianIndex, altitudes::AbstractVector) = [CartesianIndex(a[1], a[2], h) for h in altitudes]
expand_action_to_other_altitudes(a::CartesianIndex, altitudes::AbstractVector) = [CartesianIndex(a[1], a[2], h) for h in setdiff(Set(altitudes),a[3])]
expand_action_to_below_altitudes(a::CartesianIndex, altitudes::AbstractVector) = [CartesianIndex(a[1], a[2], h) for h in altitudes if h <= a[3]]

function expand_action_to_limits(a::CartesianIndex, altitudes, grid_dist, delta)
    """ Creates an array of blocked locations (considering both altitude and delta limits), given an action. """
    a1, a2 = a[1], a[2]
    blocked = [CartesianIndex(a1+δ1, a2+δ2, h) for h in altitudes for δ1 in -delta:grid_dist:delta for δ2 in -delta:grid_dist:delta]
    return blocked
end

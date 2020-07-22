struct WindFarmState
    x_acts::AbstractArray   # actions taken so far.
    x_obs::AbstractArray    # locations observed so far.
    y_obs::AbstractArray    # values of locations observed so far.
end

struct WindFarmBelief
    x_acts::AbstractArray
    gpla_wf::GPLA
end

struct WindFarmBeliefUpdater <: POMDPs.Updater
    altitudes::AbstractVector
    grid_dist::Int
end

function POMDPs.update(bu::WindFarmBeliefUpdater, old_b::WindFarmBelief, a::CartesianIndex{3}, obs::AbstractVector)
    a0 = CartIndices_to_Vector(a)
    a = expand_action_to_altitudes(a, bu.altitudes)
    a = CartIndices_to_Array(a)

    x_acts = hcat(old_b.x_acts, a0)

    gpla_wf = deepcopy(old_b.gpla_wf)
    x_obs, y_obs = gpla_wf.x, gpla_wf.y
    
    x_obs = hcat(x_obs, a)
    y_obs = vcat(y_obs, obs)
    GaussianProcesses.fit!(gpla_wf, x_obs, y_obs)

    # println("Belief Updated!")
    return WindFarmBelief(x_acts, gpla_wf)
end

@with_kw struct WindFarmBeliefInitializerParams
    # Parsing Farm Data
    farm = "AltamontCA"
    grid_dist = 220
    altitudes = [100, 150, 200]
    nx = 90
    ny = nx

    # Observation Set
    grid_dist_obs = grid_dist .* 10

    # GPLA
    num_neighbors = 5
    theta = [-1.5802417326559162, 4.0, 7.147412061026513, 0.7893190206886835, 0.3038066734614207, 1.0, 0.0, 1.3127192001252717e-209]

end

function initialize_belief_rollout(s::WindFarmState)

    # Load prior points for belief GP
    X_obs, Y_obs = s.x_obs, s.y_obs

    # Create initial kernel
    kernel = WLK_SEIso(eps(), eps(), eps(), eps(), eps(), eps())
    gpla_wf = GPLA(X_obs, Y_obs, wfparams.num_neighbors, 0, 0, MeanConst(wfparams.theta[2]), kernel, wfparams.theta[1])
    GaussianProcesses.set_params!(gpla_wf, wfparams.theta)

    x_acts = reshape(Float64[],3,0)
    return WindFarmBelief(x_acts, gpla_wf)
end

function initialize_belief_sparse(wfparams::WindFarmBeliefInitializerParams)

    # Load prior points for belief GP
    Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
    X_obs, _ = get_dataset(Map, wfparams.altitudes, wfparams.grid_dist_obs, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)
    Y_obs = map(lambda -> get_Y_from_farm_location(lambda, Map, wfparams.grid_dist), collect(eachcol(X_obs)))

    # Create initial kernel
    kernel = WLK_SEIso(eps(), eps(), eps(), eps(), eps(), eps())
    gpla_wf = GPLA(X_obs, Y_obs, wfparams.num_neighbors, 0, 0, MeanConst(wfparams.theta[2]), kernel, wfparams.theta[1])
    GaussianProcesses.set_params!(gpla_wf, wfparams.theta)

    x_acts = reshape(Float64[],3,0)
    return WindFarmBelief(x_acts, gpla_wf)
end

function initialize_belief_noisy(wfparams::WindFarmBeliefInitializerParams, windNoise::Number)

    # Load prior points for belief GP
    Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
    X_obs, _ = get_dataset(Map, wfparams.altitudes, wfparams.grid_dist, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)

    # Additional noise is added to prior observations
    noise = Distributions.Uniform(-windNoise, +windNoise)
    Y_obs = map(lambda -> get_Y_from_farm_location(lambda, Map, wfparams.grid_dist) + rand(noise), collect(eachcol(X_obs)))

    # Create initial kernel
    kernel = WLK_SEIso(eps(), eps(), eps(), eps(), eps(), eps())
    gpla_wf = GPLA(X_obs, Y_obs, wfparams.num_neighbors, 0, 0, MeanConst(wfparams.theta[2]), kernel, wfparams.theta[1])
    GaussianProcesses.set_params!(gpla_wf, wfparams.theta)

    gpla_wf.logNoise.value = log(windNoise)

    x_acts = reshape(Float64[],3,0)
    return WindFarmBelief(x_acts, gpla_wf)
end

function initialize_belief_no_prior(wfparams::WindFarmBeliefInitializerParams)
    # Load prior points for belief GP
    X_obs = reshape(Float64[],3,0)
    Y_obs = Float64[]

    # Create initial kernel
    kernel = WLK_SEIso(eps(), eps(), eps(), eps(), eps(), eps())
    gpla_wf = GPLA(X_obs, Y_obs, wfparams.num_neighbors, 0, 0, MeanConst(wfparams.theta[2]), kernel, wfparams.theta[1])
    GaussianProcesses.set_params!(gpla_wf, wfparams.theta)

    x_acts = reshape(Float64[],3,0)
    return WindFarmBelief(x_acts, gpla_wf)
end

function initialize_state(wfparams::WindFarmBeliefInitializerParams)
    """ Called only once when the initial state is created. """
    Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
    # y_hat = rand(b.gpla_wf)
    # x_obs = b.gpla_wf.x
    x_acts = reshape(Float64[],3,0)
    X_field, Y_field = get_dataset(Map, wfparams.altitudes, wfparams.grid_dist, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)
    return WindFarmState(x_acts, X_field, Y_field)
end

function POMDPs.rand(rng::AbstractRNG, b::WindFarmBelief)
    """ Called everytime a state is sampled from the belief. """
    y_hat = rand(b.gpla_wf)
    x_obs = b.gpla_wf.x
    # Map = Dict(x => y_hat[i] for (i, x) in enumerate(eachcol(x_obs)))
    return WindFarmState(b.x_acts, x_obs, y_hat)
end

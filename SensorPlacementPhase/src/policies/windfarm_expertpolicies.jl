"""
    WindFarmUCBPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater}
an expert policy that is used to select the greediest actions with respect to some Upper Confidence Bounds.

Constructor:

    `WindFarmUCBPolicy(problem::Union{POMDP,MDP};
             rng=Random.GLOBAL_RNG,
             updater=POMDPPolicies.NothingUpdater())`

# Fields 
- `rng::RNG` a random number generator 
- `probelm::P` the POMDP or MDP problem 
- `updater::U` a belief updater (default to `POMDPPolicies.NothingUpdater` in the above constructor)
"""

mutable struct WindFarmUCBPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    rng::RNG
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
WindFarmUCBPolicy(problem::Union{POMDP,MDP}; rng=Random.GLOBAL_RNG, updater=POMDPPolicies.NothingUpdater()) = WindFarmUCBPolicy(rng, problem, updater)


function greedyUCBPolicy(gpla_wf::GPLA, legal_actions::AbstractArray)
    legal_actions = CartIndices_to_Array(legal_actions)

    μ, σ² = GaussianProcesses.predict_f(gpla_wf, legal_actions)
    σ = sqrt.(σ²)
    N = length(gpla_wf.y)

    z_value = 1.645   # chosen: 90 percent confidence interval
    UCB = μ + z_value / sqrt(N) * σ
    
    best_val = argmax(vec(UCB))
    best_action = legal_actions[:,best_val]
    return best_action
end

function POMDPPolicies.action(policy::WindFarmUCBPolicy, b::WindFarmBelief)
    gpla_wf = b.gpla_wf
    legal_actions = actions(policy.problem, b)
    policy_action = greedyUCBPolicy(gpla_wf, legal_actions)                      # deterministically choose the best location.
    # @show policy_action
    return Vector_to_CartIndices(policy_action)
end



"""
    WindFarmRolloutPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater}
an expert policy that is used to select from a set of high-valued actions with respect to some expert knowledge function.

Constructor:

    `WindFarmRolloutPolicy(problem::Union{POMDP};
             rng=Random.GLOBAL_RNG,
             updater=WindFarmRolloutUpdater(POMDP))`

# Fields 
- `rng::RNG` a random number generator 
- `probelm::P` the POMDP or MDP problem 
- `updater::U` a belief updater (default to `POMDPPolicies.NothingUpdater` in the above constructor)
"""

struct WindFarmRolloutUpdater <: POMDPs.Updater
    altitudes::AbstractVector
    grid_dist::Int
end

mutable struct WindFarmRolloutPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    rng::RNG
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
WindFarmRolloutPolicy(problem::Union{POMDP,MDP}; rng=Random.GLOBAL_RNG, updater=WindFarmRolloutUpdater(problem.altitudes, problem.grid_dist)) = WindFarmRolloutPolicy(rng, problem, updater)


function rolloutExpertPolicy(gpla_wf_rollout::GPLA, legal_actions::AbstractArray)
    top_n_to_consider = 15

    legal_actions = CartIndices_to_Array(legal_actions)

    μ, σ² = GaussianProcesses.predict_f(gpla_wf_rollout, legal_actions)
    σ = sqrt.(σ²)
    N = length(gpla_wf_rollout.y)

    z_value = 1.645   # chosen: 90 percent confidence interval
    UCB = μ + z_value / sqrt(N) * σ
    # UCB = dropdims(μ, dims=2)

    
    best_vals = partialsortperm(vec(UCB), 1:top_n_to_consider, rev=true)

    best_actions = legal_actions[:,best_vals]
    weights = UCB[best_vals]
    return best_actions, weights
end

function POMDPPolicies.action(policy::WindFarmRolloutPolicy, b::WindFarmBelief)
    gpla_wf_rollout = b.gpla_wf
    legal_actions = actions(policy.problem, b)
    best_actions, weights = rolloutExpertPolicy(gpla_wf_rollout, legal_actions)
    # @show weights
    policy_action = StatsBase.sample(collect(eachcol(best_actions)), Weights(weights))        # sample one candidate action w.r.t weights.
    return Vector_to_CartIndices(policy_action)
end

function POMDPs.update(bu::WindFarmRolloutUpdater, old_b::WindFarmBelief, a::CartesianIndex{3}, obs::AbstractVector)
    a0 = CartIndices_to_Vector(a)
    a = expand_action_to_below_altitudes(a, bu.altitudes)
    a = CartIndices_to_Array(a)

    x_acts = hcat(old_b.x_acts, a0)

    # gpla_wf = deepcopy(old_b.gpla_wf)
    gpla_wf = old_b.gpla_wf

    x_obs, y_obs = gpla_wf.x, gpla_wf.y
    
    x_obs = hcat(x_obs, a)
    y_obs = vcat(y_obs, obs)
    GaussianProcesses.fit!(gpla_wf, x_obs, y_obs)

    # println("Rollout Belief Updated!")
    return WindFarmBelief(x_acts, gpla_wf)
end


## convenience functions ##
POMDPs.updater(policy::WindFarmRolloutPolicy) = policy.updater
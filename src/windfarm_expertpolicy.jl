### WindFarmExpertPolicy ###
using StatsBase
"""
    WindFarmExpertPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater}
a generic policy that uses the actions function to create a list of actions and then randomly samples an action from it.

Constructor:

    `WindFarmExpertPolicy(problem::Union{POMDP,MDP};
             rng=Random.GLOBAL_RNG,
             updater=POMDPPolicies.NothingUpdater())`

# Fields 
- `rng::RNG` a random number generator 
- `probelm::P` the POMDP or MDP problem 
- `updater::U` a belief updater (default to `POMDPPolicies.NothingUpdater` in the above constructor)
"""

struct testUpdater <: POMDPs.Updater
    grid_dist::Int
end

mutable struct WindFarmExpertPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    rng::RNG
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
WindFarmExpertPolicy(problem::Union{POMDP,MDP}; rng=Random.GLOBAL_RNG, updater=testUpdater(problem.grid_dist)) = WindFarmExpertPolicy(rng, problem, updater)

## policy execution ##
function rolloutExpertPolicy(gpla_wf_rollout::GPLA, legal_actions::AbstractArray)
    λ = 1.0    # TODO: Coefficient for estimating profit.
    no_of_turbines = 10   # TODO: Change this?

    legal_actions = CartIndices_to_Array(legal_actions)

    μ, σ² = GaussianProcesses.predict_f(gpla_wf_rollout, legal_actions)
    σ = sqrt.(σ²)
    N = length(σ)

    # z_value = 1.645   # chosen: 90 percent confidence interval
    # UCB = dropdims(μ, dims=2) + z_value / sqrt(N) * dropdims(σ, dims=2)
    UCB = dropdims(μ, dims=2)

    
    best_vals = partialsortperm(vec(UCB), 1:no_of_turbines, rev=true)
    # expected_profit = λ * sum(μ[best_vals] .^3)

    best_actions = legal_actions[:,best_vals]
    weights = UCB[best_vals]
    return best_actions, weights
end

function POMDPPolicies.action(policy::WindFarmExpertPolicy, s::WindFarmState)
    gpla_wf_rollout = get_GPLA_for_gen(s, wfparams)
    legal_actions = actions(policy.problem, s)
    best_actions, weights = rolloutExpertPolicy(gpla_wf_rollout, legal_actions)
    @show weights
    policy_action = sample(collect(eachcol(best_actions)), Weights(weights))
    return Vector_to_CartIndices(policy_action)
end

# function POMDPPolicies.action(policy::WindFarmExpertPolicy, b::WindFarmBelief)
#     gpla_wf = b.gpla_wf
#     legal_actions = actions(policy.problem, b)
#     best_actions, weights = rolloutExpertPolicy(gpla_wf, legal_actions)
#     @show weights
#     policy_action = sample(collect(eachcol(best_actions)), Weights(weights))
#     return Vector_to_CartIndices(policy_action)
# end

function POMDPPolicies.action(policy::WindFarmExpertPolicy, b::WindFarmBelief)
    gpla_wf = b.gpla_wf
    legal_actions = actions(policy.problem, b)
    best_actions, weights = rolloutExpertPolicy(gpla_wf, legal_actions)
    # @show weights
    policy_action = best_actions[:,1]    # deterministically choose the best location.
    return Vector_to_CartIndices(policy_action)
end

# function POMDPPolicies.action(policy::WindFarmExpertPolicy, b::Nothing)
#     @show 1
#     return rand(policy.rng, POMDPPolicies.actions(policy.problem))
# end

## convenience functions ##
POMDPPolicies.updater(policy::WindFarmExpertPolicy) = policy.updater
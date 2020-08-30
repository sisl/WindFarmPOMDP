using Statistics

"""
    MutualInfoPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater}
an expert policy that is used to select the greediest actions with respect to some Upper Confidence Bounds.

Constructor:

    `MutualInfoPolicy(problem::Union{POMDP,MDP};
             rng=Random.GLOBAL_RNG,
             updater=POMDPPolicies.NothingUpdater())`

# Fields 
- `rng::RNG` a random number generator 
- `probelm::P` the POMDP or MDP problem 
- `updater::U` a belief updater (default to `POMDPPolicies.NothingUpdater` in the above constructor)
"""

mutable struct MutualInfoPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    rng::RNG
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
MutualInfoPolicy(problem::Union{POMDP,MDP}; rng=Random.GLOBAL_RNG, updater=POMDPPolicies.NothingUpdater()) = MutualInfoPolicy(rng, problem, updater)


"""
function greedyMutualInfoPolicy_approx(gpla_wf::GPLA, legal_actions::AbstractArray, pomdp::POMDP)
    legal_actions = CartIndices_to_Array(legal_actions)
    
    conditional_entropy(σ²) = 0.5 * log(2 * pi * exp(1) * σ²)


    #### Part 1 ###

    function conditional_entropy_of_action1(gpla_wf, a)
        
        μ, σ² = GaussianProcesses.predict_f(gpla_wf, a)
        σ² = σ²[1]    # convert 1×1 Array to Number.
        return conditional_entropy(σ²)
    end
    
    conditional_entropy_of_actions = conditional_entropy_of_action1.(Ref(gpla_wf), eachcol(legal_actions))


    #### Part 2 ###

    function get_complement_actions(gpla_wf, pomdp, a)
        
        # Set{CartesianIndex} of all actions.
        all_actions_Set = Set(POMDPs.actions(pomdp))

        # Set{CartesianIndex} of the previous actions.
        prev_actions_Set = Vector_to_CartIndices.(eachcol(gpla_wf.x))

        # Set{CartesianIndex} of the actions that will be observed if `a` is taken.
        expanded = expand_action_to_below_altitudes(Vector_to_CartIndices(a), pomdp.altitudes)
        
        setdiff!(all_actions_Set, prev_actions_Set)
        setdiff!(all_actions_Set, expanded)
        
        complements = CartIndices_to_Array(collect(all_actions_Set))
        return complements
    end
    
    function conditional_entropy_of_complement_action1(gpla_wf, pomdp, a)
        
        # Create temp kernel 
        complements = get_complement_actions(gpla_wf, pomdp, a)
        y_hat = zeros(size(complements, 2))
        gpla_temp = GPLA(complements, y_hat, wfparams.num_neighbors, 0, 0, gpla_wf.mean, gpla_wf.kernel, gpla_wf.logNoise.value)

        μ, σ² = GaussianProcesses.predict_f(gpla_temp, a)
        σ² = σ²[1]    # convert 1×1 Array to Number.
        return conditional_entropy(σ²)
    end
    
    results = []

    Threads.@threads for idx in 1:size(legal_actions, 2)
        a = legal_actions[:,idx]
        entro = conditional_entropy_of_complement_action1(gpla_wf, pomdp, a)
        @show (idx,entro)
        push!(results, (idx,entro))
    end

    ### Combining Part 1 and 2 ###

    conditional_entropy_of_complement_actions = [item[2] for item in sort(results)]
    conditional_entropy_of_complement_actions = round.(conditional_entropy_of_complement_actions; digits=10)
    
    global ll = legal_actions
    @show size(conditional_entropy_of_actions)
    @show size(conditional_entropy_of_complement_actions)
    mutual_informations = conditional_entropy_of_actions - conditional_entropy_of_complement_actions

    best_vals = argmaxall(mutual_informations; threshold = 0.0)

    # if more than one best action, choose the one closest to the centroid of legal actions.
    if length(best_vals) == 1
        return legal_actions[:, best_vals]
    else
        best_actions = legal_actions[:, best_vals]
        centroid = Statistics.mean(legal_actions, dims=2)
        dists_to_centroid = euclidean_dist.(eachcol(best_actions), Ref(centroid))

        closest_action = best_actions[:, argmin(dists_to_centroid)]
        closest_action[end] = wfparams.altitudes[end]
        return closest_action
    end
end
"""

function greedyMutualInfoPolicy(gpla_wf::GPLA, legal_actions::AbstractArray, pomdp::POMDP)
    legal_actions = CartIndices_to_Array(legal_actions)
    
    conditional_entropy(σ²) = 0.5 * log(2 * pi * exp(1) * σ²)

    ### After taking the new action ###

    function conditional_entropy_of_complement_action1(gpla_wf, pomdp, a)
        
        # Create temp kernel 
        expanded = CartIndices_to_Array(expand_action_to_below_altitudes(Vector_to_CartIndices(a), pomdp.altitudes))
        complements = hcat(gpla_wf.x, expanded)
        y_hat = zeros(size(complements, 2))
        gpla_temp = GPLA(complements, y_hat, wfparams.num_neighbors, 0, 0, gpla_wf.mean, gpla_wf.kernel, gpla_wf.logNoise.value)

        all_actions = CartIndices_to_Array(POMDPs.actions(pomdp))
        μ, σ² = GaussianProcesses.predict_f(gpla_temp, all_actions)
        return sum(conditional_entropy.(σ²))
    end


    parallel_results = []

    Threads.@threads for idx in 1:size(legal_actions, 2)
        a = legal_actions[:,idx]
        entro = conditional_entropy_of_complement_action1(gpla_wf, pomdp, a)
        @show (idx,entro)
        push!(parallel_results, (idx,entro))
    end

    conditional_entropy_of_complement_actions = [item[2] for item in sort(parallel_results)]    

    mutual_informations =  - conditional_entropy_of_complement_actions
    best_vals = argmaxall(mutual_informations; threshold = 0.0)

    # if more than one best action, choose the one closest to the centroid of legal actions.
    if length(best_vals) == 1
        return legal_actions[:, best_vals]
    else
        best_actions = legal_actions[:, best_vals]
        centroid = Statistics.mean(legal_actions, dims=2)
        dists_to_centroid = euclidean_dist.(eachcol(best_actions), Ref(centroid))

        closest_action = best_actions[:, argmin(dists_to_centroid)]
        closest_action[end] = wfparams.altitudes[end]
        return closest_action
    end
end

function POMDPPolicies.action(policy::MutualInfoPolicy, b::WindFarmBelief)
    pomdp = policy.problem
    gpla_wf = b.gpla_wf
    legal_actions = actions(pomdp, b)
    policy_action = greedyMutualInfoPolicy(gpla_wf, legal_actions, pomdp)                      # deterministically choose the best location.
    @show policy_action
    return Vector_to_CartIndices(policy_action)
end

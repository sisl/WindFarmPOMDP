"""
    POMCPOWPlanner{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater}
"""
function POMCPOWPlanner(actpolicy)

    actpolicy_dict = Dict(
        :UCB        => POMCPOWPlanner_UCB,
        :MI         => POMCPOWPlanner_MI
    )

    act_pl = actpolicy_dict[actpolicy]
    return act_pl
end

function POMCPOWPlanner_UCB(pomdp; tree_queries = 2)          # TODO: Parametrize tree_queries.

    rollout_policy = UCBRolloutPolicy(pomdp)

    solver = POMCPOWSolver(tree_queries=tree_queries,
                           check_repeat_obs=true, 
                           check_repeat_act=true, 
                           k_action=3.0,
                           alpha_action=0.3,
                           k_observation=3.0, 
                           alpha_observation=0.3,
                           criterion = MaxUCB(1.0),
                           next_action = UCBWideningPolicy,
                           estimate_value=POMCPOW.RolloutEstimator(rollout_policy)
    )

    planner = solve(solver, pomdp)
    return planner
end

function POMCPOWPlanner_MI(pomdp; tree_queries = 2)          # TODO: Parametrize tree_queries.

    rollout_policy = MIRolloutPolicy(pomdp)

    solver = POMCPOWSolver(tree_queries=tree_queries,
                           check_repeat_obs=true, 
                           check_repeat_act=true, 
                           k_action=3.0,
                           alpha_action=0.3,
                           k_observation=3.0, 
                           alpha_observation=0.3,
                           criterion = MaxUCB(1.0),
                           next_action = rollout_policy,
                           estimate_value=POMCPOW.RolloutEstimator(rollout_policy)
    )

    planner = solve(solver, pomdp)
    return planner
end

function BasicPOMCP.extract_belief(bu::MCTSRolloutUpdater, node::BeliefNode)
    # global GNode = node
    s = rand(node.tree.sr_beliefs[end].dist)[1]    # rand simply extracts here. it is deterministic.
    return initialize_belief_rollout(s)
end
function extract_solver_method(pomdp, solvermethod, actpolicy=:UCB)
    policy_dict = Dict(
        :entropy        => ShannonEntropyPolicy,
        :mutualinfo     => MutualInfoPolicy,
        :diffentro      => DiffEntroPolicy,
        # :bayesian       => BayesianPlanner,        # TODO.
        :genetic        => GeneticPlanner,
        :pomcpow        => POMCPOWPlanner(actpolicy)
    )

    return policy_dict[solvermethod](pomdp)
end

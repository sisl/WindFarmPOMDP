function extract_solver_method(pomdp, solvermethod, extra_params)
    policy_dict = Dict(
        :entropy        => ShannonEntropyPolicy,
        :mutualinfo     => MutualInfoPolicy,
        :diffentro      => DiffEntroPolicy,
        # :bayesian       => BayesianPlanner,        # TODO.
        :genetic        => GeneticPlanner,
        :pomcpow        => POMCPOWPlanner
    )

    return policy_dict[solvermethod](pomdp, extra_params)
end

extract_solver_method(pomdp, solvermethod) = extract_solver_method(pomdp, solvermethod, [])

function get_func_name(solvermethod)
    script_dict = Dict(
        :entropy        => "./solve_SensorPP_greedy.jl",
        :mutualinfo     => "./solve_SensorPP_greedy.jl",
        :diffentro      => "./solve_SensorPP_greedy.jl",
        # :bayesian       => "./solve_SensorPP_bayesian.jl",        # TODO.
        :genetic        => "./solve_SensorPP_genetic.jl",
        :pomcpow        => "./solve_SensorPP_pomcpow.jl"
    )

    return script_dict[solvermethod]
end
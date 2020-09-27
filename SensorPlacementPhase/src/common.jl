"""
    Common functions used in the Sensor Placement Phase.
"""

function extract_solver_method(pomdp, solvermethod, extra_params)
    policy_dict = Dict(
        :entropy        => ShannonEntropyPolicy,
        :mutualinfo     => MutualInfoPolicy,
        :diffentro      => DiffEntroPolicy,
        :greedy         => UCBGreedyPolicy,
        :random         => RandomPlanner,
        :bayesian       => BayesianPlanner,
        :genetic        => GeneticPlanner,
        :pomcpow        => POMCPOWPlanner
    )

    return policy_dict[solvermethod](pomdp, extra_params)
end

extract_solver_method(pomdp, solvermethod) = extract_solver_method(pomdp, solvermethod, [])

# function get_func_name(solvermethod)
#     script_dict = Dict(
#         :entropy        => "./solve_SensorPP_greedy.jl",
#         :mutualinfo     => "./solve_SensorPP_greedy.jl",
#         :diffentro      => "./solve_SensorPP_greedy.jl",
#         :bayesian       => "./solve_SensorPP_bayesian.jl",
#         :genetic        => "./solve_SensorPP_genetic.jl",
#         :pomcpow        => "./solve_SensorPP_pomcpow.jl"    TODO: add other solvers, if any.
#     )
#
#     return script_dict[solvermethod]
# end
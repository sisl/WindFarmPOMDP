include("./policies/diffentro_sensorpolicy.jl")
include("./policies/mutualinfo_sensorpolicy.jl")
include("./policies/shannon_sensorpolicy.jl")
include("./policies/windfarm_expertpolicies.jl")

include("./policies/pomcpow_sensorplanner.jl")


function extract_solver_method(pomdp, solvermethod)
    policy_dict = Dict(
        :entropy        => ShannonEntropyPolicy,
        :mutualinfo     => MutualInfoPolicy,
        :diffentro      => DiffEntroPolicy,
        # :bayesian       => BayesianPolicy,        # TODO.
        # :genetic        => GeneticPolicy,         # TODO.
        :pomcpow        => POMCPOWPolicy
    )

    return policy_dict[solvermethod](pomdp)
end
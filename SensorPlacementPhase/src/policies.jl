include("./policies/diffentro_sensorpolicy.jl")
include("./policies/mutualinfo_sensorpolicy.jl")
include("./policies/shannon_sensorpolicy.jl")
include("./policies/windfarm_expertpolicies.jl")

function extract_policy_method(pomdp, solvermethod)
    policy_dict = Dict(
        :entropy        => ShannonEntropyPolicy,
        :mutualinfo     => MutualInfoPolicy,
        :diffentro      => DiffEntroPolicy
    )

    return policy_dict[solvermethod](pomdp)
end
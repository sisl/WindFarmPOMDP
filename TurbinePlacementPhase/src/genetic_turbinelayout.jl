"""
    Genetic method to heuristically determine optimal turbine layout.
"""

struct GeneticTurbineLayout <: TurbineLayoutType end

function turbine_profit(locs , X_field, gpla_wf, tlparams)

    x_turbines = X_field[:, locs]
    μ, _ = GaussianProcesses.predict_f(gpla_wf, x_turbines)
    cost = get_turbine_cost.(eachcol(x_turbines))
    power = get_power_production.(μ, Ref(tlparams))

    result = sum(power .- cost)
    # @show result
    return Int(round(result))
end

function get_turbine_layout(gpla_wf::GPLA, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams, layouttype::GeneticTurbineLayout)
    
    no_of_turbines = tlparams.no_of_turbines
    X_field = CartIndices_to_Array(turbine_action_space(tlparams, wfparams))

    function init_locs()
        _, loc = get_random_init_solution(X_field, no_of_turbines, tlparams)
        return loc
    end

    lx = fill(1, no_of_turbines)
    ux = fill(size(X_field, 2), no_of_turbines)
    tc = fill(Int, no_of_turbines)

    function cons(locs, X_field)
        x_turbines = X_field[:, locs]
        return [ sum(is_solution_separated_Int(x_turbines, tlparams)) ]
    end

    lc, uc = [0.0], [0.0]

    cb = Evolutionary.ConstraintBounds(lx,ux,lc,uc)
    constraints = MixedTypePenaltyConstraints(PenaltyConstraints(1e3, cb, x -> cons(x, X_field)), tc)

    opts = Evolutionary.Options(iterations=1000, abstol=1e-5)
    mthd = GA(populationSize=1000, crossoverRate=0.8, mutationRate=0.1, selection=sus, crossover=Evolutionary.uniform)


    GA_result = Evolutionary.optimize(x -> - turbine_profit(x, X_field, gpla_wf, tlparams),
                                constraints,
                                init_locs,
                                mthd,
                                opts
    )
    
    x_turbines = X_field[:, GA_result.minimizer]
    expected_profit = - GA_result.minimum
    return x_turbines, expected_profit
end
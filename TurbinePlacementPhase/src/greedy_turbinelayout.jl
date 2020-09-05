"""
    Greedy method to heuristically determine optimal turbine layout.
"""

struct GreedyTurbineLayout <: TurbineLayoutType end

function get_turbine_layout(gpla_wf::GPLA, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams, layouttype::GreedyTurbineLayout)
    
    no_of_turbines = tlparams.no_of_turbines
    X_field = CartIndices_to_Array(turbine_action_space(tlparams, wfparams))
    
    x_turbines = reshape(Float64[], 3, 0)
    expected_profit = Float64[]

    while no_of_turbines > 0
        X_field = remove_seperated_locations(x_turbines, X_field, tlparams)

        next_turbine, next_profit = get_next_turbine_location(gpla_wf, X_field, tlparams, layouttype)

        x_turbines = hcat(x_turbines, next_turbine)
        push!(expected_profit, next_profit)

        no_of_turbines = no_of_turbines - 1
    end

    return x_turbines, expected_profit
end

function get_next_turbine_location(gpla_wf, X_field, tlparams, layouttype::GreedyTurbineLayout)

    μ, σ² = GaussianProcesses.predict_f(gpla_wf, X_field)
    σ = sqrt.(σ²)
    N = length(gpla_wf.y)

    z_value = 1.645    # chosen: 90 percent confidence interval
    LCB = μ - z_value / sqrt(N) * σ
    best_val = argmax(vec(LCB))

    next_turbine = X_field[:, best_val]
    next_profit = get_power_production(vec(LCB)[best_val], tlparams)

    return next_turbine, next_profit
end
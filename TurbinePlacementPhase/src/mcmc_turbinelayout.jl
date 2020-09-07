"""
    MCMC method to heuristically determine optimal turbine layout.
"""

struct MCMCTurbineLayout <: TurbineLayoutType end

function get_turbine_layout(gpla_wf::GPLA, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams, layouttype::MCMCTurbineLayout)

    no_of_turbines = tlparams.no_of_turbines
    X_field = CartIndices_to_Array(turbine_action_space(tlparams, wfparams))

    
    no_of_trials = 100
    no_of_iterations = 100    # per trial
    obj_func = x -> turbine_approximate_profit(x, X_field, gpla_wf, tlparams)
    
    best_state = reshape(Float64[], 3, 0)
    best_state_val = 0.0
    
    while no_of_trials > 0
        
        init_state, _ = get_random_init_solution(X_field, no_of_turbines, tlparams)
        next_state = mcmc(init_state,
                        obj_func,
                        no_of_iterations,
                        X_field,
                        tlparams
        )

        next_state_val = obj_func(next_state)
        best_state_val ≤ next_state_val ? (best_state, best_state_val) = (copy(next_state), copy(next_state_val)) : nothing
        
        # @show no_of_trials, obj_func(init_state), obj_func(best_state)
        no_of_trials = no_of_trials - 1
    end

    expected_profit = obj_func(best_state)
    return x_turbines, expected_profit
end
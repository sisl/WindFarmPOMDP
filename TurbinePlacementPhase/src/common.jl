"""
    Common functions used in the Turbine Placement Phase.
"""

abstract type TurbineLayoutType end

@with_kw struct TurbineLayoutParams
    # Layout params
    layouttype::TurbineLayoutType
    grid_dist = 220                                             # [meters]
    altitudes = [100, 200]                                      # [meters]

    # Wind unit-direction
    wind_direction = [1,1,0]

    # Sensor tower specs
    mast_cost = 2.0e3                                           # [USD/meter]

    # Turbine specs
    no_of_turbines = 10
    turbine_cost = 4.0e6                                        # [USD/turbine]
    turbine_diameter = 120                                      # [meters]
    turbine_max_power = 2500                                    # [kW]
    turbine_cut_in_speed = 3.0                                  # [m/s]
    turbine_rated_speed = 11.0                                  # [m/s]
    turbine_power_curve::Polynomial = create_power_curve()      # [kW, given windspeed]
end

function create_power_curve(poly_degree::Int = 3, data_dir = "../../TurbinePlacementPhase/data/ge_turbine_data.csv")
    power_data = DelimitedFiles.readdlm(data_dir, Float64, comments=true, comment_char='#')
    speed_ms, power_kW = eachcol(power_data)
    curve_fit = Polynomials.fit(speed_ms, power_kW, poly_degree)
    return curve_fit
    # scatter(speed_ms, power_kW, label="Data")
    # plot!(curve_fit, extrema(speed_ms)..., label="Fit")
end

function TurbineLayoutParams(layoutfinder::Symbol)
    layout_dict = Dict(
        :greedy         => GreedyTurbineLayout,
        :genetic        => GeneticTurbineLayout,
        :mcmc           => MCMCTurbineLayout        
    )

    layouttype = layout_dict[layoutfinder]()
    return TurbineLayoutParams(layouttype=layouttype)
end

function get_seperation_heuristic(xi, xj, ui, uj, turbine_diameter)
""" Find the minimum required distance for two coordinates to be out of each others' wake seperation regions. """ 
    
    # Use ηmin = 3D and ηmax = 10D, as commonly done.
    ηmin, ηmax = [3, 10] .* turbine_diameter

    if view(xi, 1:2) == view(xj, 1:2) 
        return ηmax
    else
        vij = 0.5 * (ui + uj)
        dij = xi - xj
        ηij = ηmin + (ηmax - ηmin) * norm(dot_product(dij, vij)) / norm(dij) / norm(vij)
        return ηij
    end
end

function is_solution_separated(x_turbines, tlparams)
""" Check if any point in `x_turbines` are inside another's seperation region. """ 
    ui = uj = tlparams.wind_direction
    dia = tlparams.turbine_diameter
    idx_set = collect(1:size(x_turbines, 2))

    if isempty(x_turbines)
        return false
    else
        for (idx, xi) in enumerate(eachcol(x_turbines))
            idx_complement = setdiff(idx_set, collect(1:idx))
            x_turbines_complement = x_turbines[:, idx_complement]
            η = get_seperation_heuristic.(Ref(xi), eachcol(x_turbines_complement), Ref(ui), Ref(uj), Ref(dia))
            d = euclidean_dist.(Ref(xi), eachcol(x_turbines_complement))
            all(η .< d) ? nothing : return true
        end
    end

    return false    # return false if solution is NOT separated.
end

function is_solution_separated_Int(x_turbines, tlparams)
""" Check if any point in `x_turbines` are inside another's seperation region. Returns number of placements under seperation. """ 
    ui = uj = tlparams.wind_direction
    dia = tlparams.turbine_diameter
    idx_set = collect(1:size(x_turbines, 2))

    if isempty(x_turbines)
        return 0
    else
        sep = 0
        for (idx, xi) in enumerate(eachcol(x_turbines))
            idx_complement = setdiff(idx_set, collect(1:idx))
            x_turbines_complement = x_turbines[:, idx_complement]
            η = get_seperation_heuristic.(Ref(xi), eachcol(x_turbines_complement), Ref(ui), Ref(uj), Ref(dia))
            d = euclidean_dist.(Ref(xi), eachcol(x_turbines_complement))
            sep = sep + sum(η .≥ d) 
        end
        return sep
    end
end
    
function remove_seperated_locations(X_field, x_turbines, tlparams)
""" Remove the locations in `X_field` that are effected by the seperation region of any point in `x_turbines`.
    This reduces the turbine placement action space. """

    ui = uj = tlparams.wind_direction
    dia = tlparams.turbine_diameter

    if isempty(x_turbines)
        nothing
    else
        for xi in eachcol(x_turbines)
            η = get_seperation_heuristic.(Ref(xi), eachcol(X_field), Ref(ui), Ref(uj), Ref(dia))
            d = euclidean_dist.(Ref(xi), eachcol(X_field))
            X_field = X_field[:, η .< d]
        end
        X_field = remove_same_coordinate_locations(X_field, x_turbines)
    end

    return X_field
end

function find_same_coordinate_locations(xi, X_field)
""" Find the locations in `X_field` that are in the same coordinate as `xi`. """
    result = []

    for (idx, xj) in enumerate(eachcol(X_field))
        if view(xi, 1:2) == view(xj, 1:2) 
            push!(result, idx)
        end
    end
    return result
end

function remove_same_coordinate_locations(X_field, x_turbines)
""" Remove the locations in `X_field` that are in the same coordinate with any of the locations in `x_turbines`. """
    locs = find_same_coordinate_locations.(eachcol(x_turbines), Ref(X_field))
    locs_complement = setdiff(collect(1:size(X_field, 2)), vcat(locs...))
    return X_field[:, locs_complement]
end

function get_tower_cost(a)
""" Get the cost of a single mast (sensor) tower. """
    height = a[end]
    return height * tlparams.mast_cost
end

function get_turbine_cost(a)
""" Get the cost of a single turbine. """
    return tlparams.turbine_cost
end

function turbine_action_space(tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams)
""" Get the entire action space for possible turbine placements. """
    altitudes = tlparams.altitudes
    grid_dist = tlparams.grid_dist

    x_ranges = 0 : grid_dist : grid_dist * wfparams.nx-1
    y_ranges = 0 : grid_dist : grid_dist * wfparams.ny-1
    all_actions = [CartesianIndex(x,y,z) for x in x_ranges for y in y_ranges for z in altitudes]

    return vec(collect(all_actions))
end

function get_random_init_solution(X_field, no_of_turbines, tlparams)
""" Returns a random initial valid turbine placement layout. """ 
    x_turbines_init = reshape(Float64[], 3, 0)
    is_separated = true
    locs = zeros(no_of_turbines)

    while is_separated
        locs = rand(1:size(X_field, 2), no_of_turbines)
        x_turbines_init = X_field[:, locs]
        is_separated = is_solution_separated(x_turbines_init, tlparams)
    end
    return x_turbines_init, locs
end

function get_average_turbine_distances(x_turbines)
""" Calculate the average spacing [meters] between turbines. """
    dists = Float64[]
    idx_set = collect(1:size(x_turbines, 2))

    while length(idx_set) > 1
        # @show idx_set
        idx = pop!(idx_set)
        xi = x_turbines[:, idx]
        d = euclidean_dist.(Ref(xi), eachcol(x_turbines[:,idx_set]))
        push!(dists, minimum(d))
    end   
    return mean(dists)
end

# function get_average_turbine_distances_old(x_turbines)
# """ Calculate the average spacing [meters] between turbines. """
#     dists = Float64[]
#     idx_set = collect(1:size(x_turbines, 2))
#
#     for (idx, xi) in enumerate(eachcol(x_turbines))
#         idx_complement = setdiff(idx_set, collect(1:idx))
#         x_turbines_complement = x_turbines[:, idx_complement]
#         d = euclidean_dist.(Ref(xi), eachcol(x_turbines_complement))
#         append!(dists, d)
#     end
#     return dists
# end

function get_layout_profit(sp::WindFarmState, gpla_wf::GPLA, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams, layouttype::TurbineLayoutType)
""" Calculate approximate profit of a turbine layout. """
    # Cost of sensor tower placements
    x_sensors = sp.x_acts
    cost_masts = get_tower_cost.(eachcol(x_sensors))

    # Profit of turbine placements
    x_turbines, _ = get_turbine_layout(gpla_wf, tlparams, wfparams, layouttype)
    expected_turbine_profits = turbine_approximate_profits(x_turbines, X_field, gpla_wf, tlparams)

    total_profit = sum(expected_turbine_profits) - sum(cost_masts)
    return total_profit
end

function turbine_approximate_profits(x_turbines::AbstractMatrix, gpla_wf, tlparams; penalty_cost = 2.0e6)
""" Approximate sum of profits of individual turbines. """
    μ, _ = GaussianProcesses.predict_f(gpla_wf, x_turbines)
    costs = get_turbine_cost.(eachcol(x_turbines))

    s_avg = get_average_turbine_distances(x_turbines) / tlparams.turbine_diameter
    profits = get_turbine_profit.(μ, costs, Ref(s_avg), Ref(tlparams))

    result = sum(profits) - penalty_cost * sum(is_solution_separated_Int(x_turbines, tlparams))    # Penalty added for solutions within seperated regions.
    return Int(round(result))
end

get_layout_profit(sp::WindFarmState, gpla_wf::GPLA, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams) = get_layout_profit(sp, gpla_wf, tlparams, wfparams, tlparams.layouttype)
turbine_approximate_profits(locs::AbstractVector, X_field, gpla_wf, tlparams) = turbine_approximate_profits(X_field[:, locs], gpla_wf, tlparams)

function get_turbine_profit(ui, turbine_cost, s_avg, tlparams)
""" Follows the heuristic as described in Stevens et al. """
    θ = 1.0e-3
    β = 5.0e-3
    γ = 4

    P∞ = get_turbine_power_output(ui, tlparams)
    P₁ = tlparams.turbine_max_power

    # Calculate the ratio between Revenue and Costs, for a single turbine.
    profit_star = (P∞ / P₁) * γ - (1 + β * s_avg + θ * s_avg^2)

    return profit_star * turbine_cost
end

function get_turbine_power_output(ui, tlparams)
""" Keeps the power output within reasonable limits. """
    ui_clamped = clamp(ui, tlparams.turbine_cut_in_speed, tlparams.turbine_rated_speed)
    return tlparams.turbine_power_curve(ui_clamped)
end
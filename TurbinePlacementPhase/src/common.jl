"""
    Common functions used in the Turbine Placement Phase.
"""

abstract type TurbineLayoutType end

@with_kw struct TurbineLayoutParams
    # Layout params
    layouttype::TurbineLayoutType
    grid_dist = 220                     # [meters]
    altitudes = [100, 200]              # [meters]

    # Wind unit-direction
    wind_direction = [1,1,0]

    # Turbine specs
    no_of_turbines = 10
    turbine_diameter = 75               # [meters]

    # Sensor tower costs
    permanent_mast_cost = 2.9           # [USD/m]
    temporary_mast_cost = 1.15          # [USD/m]

    # Turbine costs
    turbine_cost = 0.0                  # TODO: Change this.
end

function TurbineLayoutParams(layoutfinder::Symbol)
    layout_dict = Dict(
        :greedy        => GreedyTurbineLayout,
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
    return height * tlparams.temporary_mast_cost    # TODO: Change this?
end

function get_turbine_cost(a)
""" Get the cost of a single turbine. """
    height = a[end]
    return height * tlparams.turbine_cost
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

function get_layout_revenue(sp::WindFarmState, gpla_wf::GPLA, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams, layouttype::TurbineLayoutType)
""" Calculate approximate revenue of a turbine layout. """
    # Cost of sensor tower placements
    x_sensors = sp.x_acts
    cost_masts = get_tower_cost.(eachcol(x_sensors))

    # Cost and Profit of turbine placements
    x_turbines, expected_profit = get_turbine_layout(gpla_wf, tlparams, wfparams, layouttype)
    cost_turbines = get_turbine_cost.(eachcol(x_turbines))

    # Cost of other expenses
    cost_other = 0.0    # TODO: Change this.

    total_revenue = sum(expected_profit) - sum(cost_masts) - sum(cost_turbines) - sum(cost_other)
    return total_revenue
end

get_layout_revenue(sp::WindFarmState, gpla_wf::GPLA, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams) = get_layout_revenue(sp, gpla_wf, tlparams, wfparams, tlparams.layouttype)

function get_power_production(ui, tlparams)
""" Calculate approximate power production, given a wind speed. """
    λ = 1.0    # TODO: Coefficient for estimating profit.
    return λ * ui^3
end

function turbine_approximate_profit(locs::AbstractVector, X_field, gpla_wf, tlparams)
    x_turbines = X_field[:, locs]
    return turbine_approximate_profit(x_turbines, X_field, gpla_wf, tlparams)
end

function turbine_approximate_profit(x_turbines::AbstractMatrix, X_field, gpla_wf, tlparams)
    μ, _ = GaussianProcesses.predict_f(gpla_wf, x_turbines)
    cost = get_turbine_cost.(eachcol(x_turbines))
    power = get_power_production.(μ, Ref(tlparams))

    result = sum(power .- cost) - 1000 * sum(is_solution_separated_Int(x_turbines, tlparams))
    return Int(round(result))
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
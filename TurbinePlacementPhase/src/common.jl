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

function get_seperation_heuristic(xi, xj, ui, uj)

    # Use ηmin = 3D and ηmax = 10D, as commonly done.
    ηmin, ηmax = [3, 10] .* tlparams.turbine_diameter

    vij = 0.5 * (ui + uj)
    dij = xi - xj
    ηij = ηmin + (ηmax - ηmin) * norm(dot_product(dij, vij)) / norm(dij) / norm(vij)

    return ηij
end

function remove_seperated_locations(x_turbines, X_field, tlparams)
    ui = uj = tlparams.wind_direction

    if isempty(x_turbines)
        nothing
    else
        for xi in eachcol(x_turbines)
            η = get_seperation_heuristic.(Ref(xi), eachcol(X_field), Ref(ui), Ref(uj))
            d = euclidean_dist.(Ref(xi), eachcol(X_field))
            X_field = X_field[:, η .< d]
        end
        X_field = remove_same_coordinate_locations(X_field, x_turbines)
    end

    return X_field
end

function find_same_coordinate_locations(xi, X_field)
    result = []

    for (idx, xj) in enumerate(eachcol(X_field))
        if view(xi, 1:2) == view(xj, 1:2) 
            push!(result, idx)
        end
    end
    return result
end

function remove_same_coordinate_locations(X_field, x_turbines)
    locs = find_same_coordinate_locations.(eachcol(x_turbines), Ref(X_field))
    locs_complement = setdiff(collect(1:size(X_field, 2)), vcat(locs...))
    return X_field[:, locs_complement]
end

function get_tower_cost(a)
    height = a[end]
    return height * tlparams.temporary_mast_cost    # TODO: Change this?
end

function get_turbine_cost(a)
    height = a[end]
    return height * tlparams.turbine_cost
end

function turbine_action_space(tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams)
    altitudes = tlparams.altitudes
    grid_dist = tlparams.grid_dist

    x_ranges = 0 : grid_dist : grid_dist * wfparams.nx-1
    y_ranges = 0 : grid_dist : grid_dist * wfparams.ny-1
    all_actions = [CartesianIndex(x,y,z) for x in x_ranges for y in y_ranges for z in altitudes]

    return vec(collect(all_actions))
end

function get_layout_revenue(sp::WindFarmState, gpla_wf::GPLA, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams, layouttype::TurbineLayoutType)

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

function get_power_production(ui, tlparams)
    λ = 1.0    # TODO: Coefficient for estimating profit.
    return λ * ui^3
end
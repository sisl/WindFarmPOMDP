function save_rewards_to_disk(script_id::Symbol, rewards_history::AbstractArray, savename::String)
    println("### Total Rewards: $(sum(rewards_history)) ###")
    open(savename, "a") do io
        writedlm(io, ["Script: ", string(script_id, ".jl"), ""])
        writedlm(io, ["History: ", rewards_history..., ""])
        writedlm(io, ["Total: ", sum(rewards_history)])
    end
    return nothing
end

function plot_WindFarmPOMDP_policy!(script_id::Symbol, wfparams::WindFieldBeliefParams, actions_history::AbstractArray, rewards_history::AbstractArray, b0::WindFarmBelief)
    println("### Creating Policy Plots ###")
    nx, ny = wfparams.nx, wfparams.ny
    Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
    actions_history = CartIndices_to_Vector.(actions_history)
    
    !isdir("Figures") ? mkdir("Figures") : nothing
    dir = string("Figures/", Dates.now())
    mkdir(dir)

    gpla_wf = b0.gpla_wf
    
    for h in wfparams.altitudes

        # Extract the actions `a` taken that were at altitude `h`.
        a_in_h = reshape(Float64[],2,0)
        for a in actions_history
            if a[end]==h
                a_in_h = hcat(a_in_h, a[1:2] / wfparams.grid_dist + [1,1])    # [1,1] is added because locations were 0-based indexed.
            end
        end
        
        X_field, Y_field = get_dataset(Map, [h], wfparams.grid_dist, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)
        p = Plots.heatmap(reshape(Y_field, (nx,ny)), title="Wind Farm Sensor Locations Chosen, h = $(h)m")
        Plots.scatter!(a_in_h[2,:], a_in_h[1,:], legend=false, color=:white)  # Notice that the row and col of `a_in_h` is reversed.
        Plots.savefig(p, "./$dir/Plot_$h.png")
        Plots.savefig(p, "./$dir/Plot_$h.pdf")
    
        # Plots of initial belief below.
        μ, σ² = GaussianProcesses.predict_f(gpla_wf, X_field)
        σ = sqrt.(σ²)

        p2 = Plots.heatmap(reshape(σ, (nx,ny)), title="Wind Farm Initial Belief Variance, h = $(h)m")
        Plots.scatter!(a_in_h[2,:], a_in_h[1,:], legend=false, color=:white)  # Notice that the row and col of `a_in_h` is reversed.
        Plots.savefig(p2, "./$dir/Plot2_$h.png")
        Plots.savefig(p2, "./$dir/Plot2_$h.pdf")

        p3 = Plots.heatmap(reshape(μ, (nx,ny)), title="Wind Farm Initial Belief Mean, h = $(h)m")
        Plots.scatter!(a_in_h[2,:], a_in_h[1,:], legend=false, color=:white)  # Notice that the row and col of `a_in_h` is reversed.
        Plots.savefig(p3, "./$dir/Plot3_$h.png")
        Plots.savefig(p3, "./$dir/Plot3_$h.pdf")

    end
    save_rewards_to_disk(script_id, rewards_history, "./$dir/rewards.txt")
    
    println("### Policy Plots Saved to $dir ###")
    return nothing
end

function plot_WindFarmPOMDP_belief_history!(wfparams::WindFieldBeliefParams, actions_history::AbstractArray, belief_history::AbstractArray, b0::WindFarmBelief)
    println("### Creating Policy Plots ###")
    nx, ny = wfparams.nx, wfparams.ny
    Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
    actions_history = CartIndices_to_Vector.(actions_history)
    
    !isdir("Figures") ? mkdir("Figures") : nothing
    dir = string("Figures/", Dates.now())
    mkdir(dir)

    for (idx,b) in enumerate(vcat(b0, belief_history))
        t = idx - 1     # timestep
        for h in wfparams.altitudes

            # Extract the actions `a` taken that were at altitude `h`.
            a_in_h = reshape(Float64[],2,0)

            if idx != 1
                for a in actions_history[1:t]
                    if a[end]==h
                        a_in_h = hcat(a_in_h, a[1:2] / wfparams.grid_dist + [1,1])    # [1,1] is added because locations were 0-based indexed.
                    end
                end
            end
            
            # p = Plots.heatmap(reshape(Y_field, (nx,ny)), title="Wind Farm Sensor Locations Chosen, h = $(h)m")
            # Plots.scatter!(a_in_h[2,:], a_in_h[1,:], legend=false, color=:white)  # Notice that the row and col of `a_in_h` is reversed.
            # Plots.savefig(p, "./$dir/Plot_$h.png")
            # Plots.savefig(p, "./$dir/Plot_$h.pdf")

            # Extract field at `h`.
            X_field, Y_field = get_dataset(Map, [h], wfparams.grid_dist, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)
            
            # Plots of initial belief below.
            gpla_wf = b.gpla_wf
            μ, σ² = GaussianProcesses.predict_f(gpla_wf, X_field)
            σ = sqrt.(σ²)

            p2 = Plots.heatmap(reshape(σ, (nx,ny)), title="Variance of Belief at t=$(t), h = $(h)m")
            Plots.scatter!(a_in_h[2,:], a_in_h[1,:], legend=false, color=:white)  # Notice that the row and col of `a_in_h` is reversed.
            Plots.savefig(p2, "./$dir/PlotVar_t$(t)_h$(h).png")
            Plots.savefig(p2, "./$dir/PlotVar_t$(t)_h$(h).pdf")

            p3 = Plots.heatmap(reshape(μ, (nx,ny)), title="Mean of Belief at t=$(t), h = $(h)m")
            Plots.scatter!(a_in_h[2,:], a_in_h[1,:], legend=false, color=:white)  # Notice that the row and col of `a_in_h` is reversed.
            Plots.savefig(p3, "./$dir/PlotMean_t$(t)_h$(h).png")
            Plots.savefig(p3, "./$dir/PlotMean_t$(t)_h$(h).pdf")

        end
    end

    println("### Policy Plots Saved to $dir ###")
    return nothing
end
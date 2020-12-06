"""
    Functions for parsing cmd arguments.
"""
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--solvermethod", "-S"
            help = "Solver method for sensor placements."
            range_tester = (x -> x ∈ ["pomcpow", "greedy", "random", "entropy", "mutualinfo", "diffentro"])
            required = true

        "--layoutfinder", "-L"
            help = "Layout type for heuristically determining a turbine layout."
            range_tester = (x -> x ∈ ["greedy", "genetic", "mcmc"])
            default = "greedy"

        "--noise_seed", "-E"
            help = "Seed of the additive noise used to create the initial belief."
            arg_type = Int
            default = 123

        "--actpolicy", "-A"
            help = "The action branching & rollout policy to be used, if the `solvermethod` uses tree branching."
            range_tester = (x -> x ∈ ["UCB", "MI"])
            default = nothing

        "--tree_queries", "-T"
            help = "Number of tree queries, if the `solvermethod` uses tree branching."
            arg_type = Int
            default = nothing

        "--savename", "-N"
            help = "Save name for results. Any valid String accepted. Pass no arguments to skip saving."
            arg_type = String
            default = nothing
    end

    parsed_args = parse_args(s, as_symbols=true)
    parsed_args[:solvermethod] = Symbol(parsed_args[:solvermethod])
    parsed_args[:layoutfinder] = Symbol(parsed_args[:layoutfinder])

    return parsed_args
end

function show_args(parsed_args)
    println("## Arguments Parsed ##")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    println(" ")
end

macro show_args(parsed_args)
    return :( show_args($parsed_args) )
end



"""
    Functions for parsing solution results.
"""
function show_results_as_dataframe(csv_filenames; normalizer_rewards = 1.0e6)

    if isempty(csv_filenames) return end

    μs = []
    σs = []
    pr = []
    tt = []
    td = []
    solvername = ""

    for fl in csv_filenames
        file_path = csv_dir * fl
        parsed_data = readdlm(file_path)
        no_of_runs = size(parsed_data, 1)    # no of lines in file
        
        solvername = parsed_data[1,1]
        layouttype = parsed_data[1,2]
        params = parsed_data[1,2:end-2]
        
        times_taken = parsed_data[:, end-1]
        rewards = parsed_data[:, end] / normalizer_rewards

        avg_rewards = mean(rewards)
        std_rewards = std(rewards)./ sqrt(no_of_runs)

        avg_times = mean(times_taken)
        std_times = std(times_taken)./ sqrt(no_of_runs)

        push!(μs, avg_rewards)
        push!(σs, std_rewards)
        push!(pr, params)
        push!(tt, avg_times)
        push!(td, std_times)
    end

    # Sort all 
    srt = sortperm(pr)
    μs = μs[srt]
    σs = σs[srt]
    pr = pr[srt]
    tt = tt[srt]
    td = td[srt]

    Data_results = DataFrame(Params = pr,
    Reward = μs,
    Reward_pm = σs, 
    Time = tt,
    Time_pm = td                          
    );
    
    @show solvername
    @show Data_results
    println("\n")
end

function get_csv_filenames(files_in_dir, val)
    idx = findall(x -> occursin(val, x), files_in_dir)
    return files_in_dir[idx]
end

function parse_results(solvermethods, csv_dir)
    rd = readdir(csv_dir)

    for sv in solvermethods
        csv_filenames_sv  = get_csv_filenames(rd, String(sv))
        show_results_as_dataframe(csv_filenames_sv)
    end
end
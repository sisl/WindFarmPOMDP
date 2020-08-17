using DelimitedFiles
using Statistics
using Plots

csv_dir = "./Generic_Run_Results/"

csv_filenames_bomcp =
["solve_windfarm_bomcp_sparsebelief_2020-08-04T00:45:12.451.csv",
"solve_windfarm_bomcp_sparsebelief_2020-08-04T09:39:17.484.csv",
"solve_windfarm_bomcp_sparsebelief_2020-08-04T09:39:33.519.csv",
"solve_windfarm_bomcp_sparsebelief_2020-08-04T09:40:05.807.csv",
"solve_windfarm_bomcp_sparsebelief_2020-08-04T17:09:45.798.csv",
"solve_windfarm_bomcp_sparsebelief_2020-08-04T17:10:24.43.csv"]

csv_filenames_pomcpow =
["solve_windfarm_pomcpow_sparsebelief_2020-08-04T00:43:40.552.csv",
"solve_windfarm_pomcpow_sparsebelief_2020-08-04T09:39:25.463.csv",
"solve_windfarm_pomcpow_sparsebelief_2020-08-04T09:39:49.122.csv",
"solve_windfarm_pomcpow_sparsebelief_2020-08-04T09:40:18.324.csv",
"solve_windfarm_pomcpow_sparsebelief_2020-08-04T17:09:59.909.csv",
"solve_windfarm_pomcpow_sparsebelief_2020-08-04T17:15:19.733.csv"]

csv_filename_greedy = "solve_windfarm_greedy_sparsebelief_2020-08-16T09:40:19.807.csv"

μs_bomcp = []
σs_bomcp = []
tq_bomcp = []

## Plot BOMCP results

for fl in csv_filenames_bomcp

    file_path = csv_dir * fl
    parsed_data = readdlm(file_path)
    no_of_runs = size(parsed_data, 1)    # no of lines in file
    
    tree_queries = parsed_data[1]
    rewards = parsed_data[:, end] .* 10000

    avg_rewards = mean(rewards)
    std_rewards = std(rewards)./ sqrt(no_of_runs)

    push!(μs_bomcp, avg_rewards)
    push!(σs_bomcp, std_rewards)
    push!(tq_bomcp, tree_queries)
end

srt = sortperm(tq_bomcp)
μs_bomcp = μs_bomcp[srt]
σs_bomcp = σs_bomcp[srt]
tq_bomcp = tq_bomcp[srt]

p1 = plot(tq_bomcp, μs_bomcp, grid=true,
    ribbon = σs_bomcp, fillalpha =.5,
    label = "BOMCP",
    m = (3,:blue,:square))

xlabel!(p1, "Tree Queries")
ylabel!(p1, "Average Rewards")

yaxis!(p1, [5000, 25000])


## Plot POMCPOW results

μs_pomcpow = []
σs_pomcpow = []
tq_pomcpow = []

for fl in csv_filenames_pomcpow

    file_path = csv_dir * fl
    parsed_data = readdlm(file_path)
    no_of_runs = size(parsed_data, 1)    # no of lines in file
    
    tree_queries = parsed_data[1]
    rewards = parsed_data[:, end] .* 10000

    avg_rewards = mean(rewards)
    std_rewards = std(rewards) ./ sqrt(no_of_runs)

    push!(μs_pomcpow, avg_rewards)
    push!(σs_pomcpow, std_rewards)
    push!(tq_pomcpow, tree_queries)
end

srt = sortperm(tq_pomcpow)
μs_pomcpow = μs_pomcpow[srt]
σs_pomcpow = σs_pomcpow[srt]
tq_pomcpow = tq_pomcpow[srt]


plot!(p1, tq_pomcpow, μs_pomcpow, grid=true,
    ribbon = σs_pomcpow, fillalpha =.5,
    label = "POMCPOW",
    m = (3,:red,:square))


## Plot Greedy results
μs_greedy = []
σs_greedy = []

fl = csv_filename_greedy
file_path = csv_dir * fl
parsed_data = readdlm(file_path)
no_of_runs = size(parsed_data, 1)    # no of lines in file

rewards = parsed_data[:, end] .* 10000

avg_rewards = mean(rewards)
std_rewards = std(rewards) ./ sqrt(no_of_runs)

push!(μs_greedy, avg_rewards)
push!(σs_greedy, std_rewards)

tq_greedy = tq_bomcp
μs_greedy = repeat(μs_greedy, length(tq_greedy))
σs_greedy = repeat(σs_greedy, length(tq_greedy))

plot!(p1, tq_greedy, μs_greedy, grid=true,
    ribbon = σs_greedy, fillalpha =.5,
    label = "Expert",
    # m = (3,:green,:square),
    legendfontsize = 11,
    tickfontsize   = 12,
    guidefontsize  = 14,
    titlefontsize  = 14)


title!("Wind Farm Planning Results")


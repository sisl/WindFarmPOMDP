using DelimitedFiles
using DataFrames
using Statistics
include("../src/utils/parse_functions.jl")

csv_dir = "./Generic_Run_Results/"
rd = readdir(csv_dir)

csv_filenames_pomcpow  = get_csv_filenames(rd, "pomcpow")
csv_filenames_genetic  = get_csv_filenames(rd, "genetic")
csv_filenames_bayesian = get_csv_filenames(rd, "bayesian")

show_results_as_dataframe(csv_filenames_pomcpow)
show_results_as_dataframe(csv_filenames_genetic)
show_results_as_dataframe(csv_filenames_bayesian)
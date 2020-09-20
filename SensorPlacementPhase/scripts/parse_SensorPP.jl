using DelimitedFiles
using DataFrames
using Statistics
include("../src/utils/parse_functions.jl")

csv_dir = "./Generic_Run_Results/"
solvermethods = [:pomcpow,
                 :genetic,
                 :bayesian,
                 :greedy,
                 :random
]

parse_results(solvermethods, csv_dir)
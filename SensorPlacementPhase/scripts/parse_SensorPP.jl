using DelimitedFiles
using DataFrames
using Statistics
include("../src/utils/parse_functions.jl")

csv_dir = isempty(Main.ARGS) ? "./Generic_Run_Results/" : ARGS[1]

solvermethods = [:pomcpow,
                 :genetic,
                 :bayesian,
                 :random,
                 :greedy,
                 :entropy,
                 :mutualinfo,
                 :diffentro
]

parse_results(solvermethods, csv_dir)
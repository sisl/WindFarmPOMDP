# Modules
using POMDPs
using Parameters
using LinearAlgebra: norm

# windGP repo
include("../../../windGP/src/windGP.jl")

# TurbinePlacementPhase scripts
include("./common.jl")
include("./greedy_turbinelayout.jl")
include("./genetic_turbinelayout.jl")
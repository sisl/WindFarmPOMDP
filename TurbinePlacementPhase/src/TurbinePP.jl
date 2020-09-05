# Modules
using POMDPs
using Parameters
using LinearAlgebra: norm
using Evolutionary

# WindGP repo
include("../../../WindGP/src/WindGP.jl")

# TurbinePlacementPhase scripts
include("./common.jl")
include("./greedy_turbinelayout.jl")
include("./genetic_turbinelayout.jl")
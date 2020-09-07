# Modules
using LinearAlgebra: norm
using Parameters
using Evolutionary

# WindGP repo
include("../../../WindGP/src/WindGP.jl")

# TurbinePlacementPhase scripts
include("./utils/metropolis_hastings.jl")
include("./common.jl")
include("./greedy_turbinelayout.jl")
include("./genetic_turbinelayout.jl")
include("./mcmc_turbinelayout.jl")
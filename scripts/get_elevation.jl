# Get elevation data from SRTM database for Altamont, CA.

include("../src/windfarmpomdp.jl")
include("../src/beliefstates.jl")
include("../../windGP/src/dataparser_GWA.jl")
include("../../windGP/src/dataparser_SRTM.jl")

wfparams = WindFarmBeliefInitializerParams(nx=90, ny=90, grid_dist_obs = 220)

# Get altitude data (for comparison)
h = 100
Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
X_field, Y_field = get_dataset(Map, [h], wfparams.grid_dist_obs, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)

# Get elevation data
srtm_coord = "N37W122"
elev = get_elevation_data(wfparams.farm, srtm_coord, wfparams.grid_dist_obs, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)
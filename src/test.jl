
include("HaloModel.jl")
using HaloModel
using Plots


path = "/Volumes/workspace/data/LastJourneyProfiles_M200c_clusters_step331_bins.hdf5"

halos = HaloModel.read_data(path, "mb0", 10)
halo = halos[1]


plt = halo.plot_halos(halo)
savefig(plt, "halo.png")
println("done")





using HaloModel
using Plots


path = "/Volumes/workspace/data/LastJourneyProfiles_M200c_clusters_step331_bins.hdf5"

# wait for a keypress
halos = HaloModel.read_data(path, "mb0", 10)
halo = halos[4]


plt = HaloModel.Halos.plot_differential_profile(halo)
rho_crit = 2.77536627e11
nfw_params = HaloModel.Halos.fit_nfw(halo, rho_crit)
radii = halo.profile.radius


bin_centers = (radii[1:end-1] .+ radii[2:end]) / 2
nfw_fit = HaloModel.Halos.nfw(nfw_params[1], nfw_params[2], bin_centers, rho_crit)
plot!(plt, bin_centers, nfw_fit, label="NFW fit", lw=2)
savefig(plt, "halo.png")





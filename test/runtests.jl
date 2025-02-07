using HaloModel
using Plots


path = "/Volumes/workspace/data/LastJourneyProfiles_M200c_clusters_step331_bins.hdf5"
cosmo = HaloModel.Simulations.LastJourney.cosmology

# wait for a keypress
halos = HaloModel.read_data(path, "mb0", 10)
halo = halos[4]

plt = HaloModel.Halos.plot_differential_profile(halo)

nfw_f2 = HaloModel.Model.fit_nfw(halo, cosmo, 0.5, true)
nfw_f1 = HaloModel.Model.fit_nfw(halo, true)
rdelta = HaloModel.Halos.get_halo_property(halo, "sod_halo_radius")
mdelta = HaloModel.Halos.get_halo_property(halo, "sod_halo_mass")
radii = halo.profile.radius


bin_centers = (radii[1:end-1] .+ radii[2:end]) / 2
# Increase the size of the plot
# Move the legend to the bottom center
plot!(plt, legend=:bottom)
plot!(plt, size=(800, 600))
HaloModel.Model.plot_nfw!(plt, nfw_f1, label="NFW 1-Parameter Fit")
HaloModel.Model.plot_nfw!(plt, nfw_f2, label="NFW 2-Parameter Fit")


savefig(plt, "halo.png")





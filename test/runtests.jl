using HaloModel
using Plots


path = "/Volumes/workspace/data/LastJourneyProfiles_M200c_clusters_step331_bins.hdf5"
cosmo = HaloModel.Simulations.LastJourney.cosmology

# wait for a keypress
halos = HaloModel.read_data(path, "mb0", 10)
halo = halos[10]


nfw_f2 = HaloModel.Model.fit_nfw(halo, cosmo, 0.5)
nfw_f1 = HaloModel.Model.fit_nfw(halo)
rdelta = HaloModel.Halos.get_halo_property(halo, "sod_halo_radius")
mdelta = HaloModel.Halos.get_halo_property(halo, "sod_halo_mass")
cdelta = HaloModel.Halos.get_halo_property(halo, "sod_halo_cdelta")
radius = halo.profile.radius
bin_centers = (radius[1:end-1] .+ radius[2:end]) / 2

radii = collect(10 .^ range(log10(bin_centers[1]), log10(bin_centers[end-1]), length=100))

nfw_profile = HaloModel.Model.nfw(bin_centers, cdelta, rdelta, mdelta)

gpfit = HaloModel.GP.gp_model(halo)
_, gp_pred = HaloModel.GP.predict(halo, gpfit, radii)



mask = gp_pred .> 0
radii = radii[mask]
gp_pred = gp_pred[mask]

plt = HaloModel.Halos.plot_differential_profile(halo)
plot!(plt, radii, gp_pred, label="GP Fit")

# Increase the size of the plot
# Move the legend to the bottom center
plot!(plt, legend=:bottom)
plot!(plt, size=(800, 600))
HaloModel.Model.plot_nfw!(plt, nfw_f1, label="NFW 1-Parameter Fit")
HaloModel.Model.plot_nfw!(plt, nfw_f2, label="NFW 2-Parameter Fit")


savefig(plt, "halo.png")





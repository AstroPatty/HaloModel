import astropy.units as u
import numpy as np
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib.pyplot as plt


def make_gp(halo_properties, halo_profiles, cosmology, z_halo=0):
    bin_radius = halo_profiles["sod_halo_bin_radius"][0]
    bin_mass = halo_profiles["sod_halo_bin_mass"][0]
    bin_counts = halo_profiles["sod_halo_bin_count"][0]
    r200c = halo_properties["sod_halo_radius"]
    bound_idxs = get_radius_bound_idxs(
        halo_properties["sod_halo_radius"],
        bin_radius,
        halo_profiles["sod_halo_bin_rad_vel"][0],
        bin_counts,
    )
    bin_radius = np.insert(bin_radius, 0, 0)
    bin_centers = 0.5 * (bin_radius[1:] + bin_radius[:-1])
    bin_dmdr = bin_mass / np.diff(bin_radius)

    sigma_dmdr = get_sigma_dmdr(bin_mass, bin_counts, bin_radius)

    bin_dmdr_to_fit = bin_dmdr[bound_idxs[0] : bound_idxs[1]]
    bin_radius_to_fit = bin_centers[bound_idxs[0] : bound_idxs[1]]
    bin_sigma_dmdr_to_fit = sigma_dmdr[bound_idxs[0] : bound_idxs[1]]


    gp_regression_fit = get_gp_fit(
        bin_radius_to_fit / r200c, bin_dmdr_to_fit, bin_sigma_dmdr_to_fit
    )
    nfw_fit_params = get_nfw_fit(bin_radius_to_fit, bin_dmdr_to_fit,cosmology)

    predict_bins = np.logspace(
        np.log10(bin_radius_to_fit[0]), np.log10(bin_radius_to_fit[-1]), 1000
    )

    gp_predict_bins = predict_bins / r200c

    gp_dmdr = gp_regression_fit.predict(gp_predict_bins.reshape(-1, 1))
    nfw_dmdr = nfw(*nfw_fit_params, predict_bins, cosmology)

    gp_peak_idx = np.argmax(gp_dmdr)
    nfw_peak_idx = np.argmax(nfw_dmdr)

    gp_peak_radius = gp_predict_bins[gp_peak_idx]
    nfw_peak_radius = gp_predict_bins[nfw_peak_idx]

    c_gp = 1 / gp_peak_radius
    c_nfw = nfw_fit_params[0]

    if np.random.randint(0,50) == 7:
        c_nfw_display = round(c_nfw, 3)
        c_gp_display = round(c_gp, 3)

        plt.figure(figsize=(10,10))
        plt.plot(gp_predict_bins, gp_dmdr, label=f"C_gp = {c_gp_display:.3f}", color="blue")
        plt.plot(gp_predict_bins, nfw_dmdr, label=f"C_nfw = {c_nfw_display:.3f}", color="red")
        plt.scatter(bin_centers/r200c, bin_dmdr)
        plt.axvline(gp_peak_radius, linestyle="--", color="blue")
        plt.axvline(nfw_peak_radius, linestyle="--", color="red")
        plt.legend()
        plt.loglog()
        plt.xlabel(r"$r/r_{200c}$")
        plt.ylabel(r"$dm/dr$")
        plt.savefig(f"/home/pwells/projects/HaloModel/gps/{halo_properties['fof_halo_tag']}.png")

    return {"c_gp": c_gp, "c_nfw": c_nfw}


def get_sigma_dmdr(bin_masses, bin_counts, bin_radius):
    mp = np.sum(bin_masses) / np.sum(bin_counts)
    return mp * np.sqrt(bin_counts) / np.diff(bin_radius)


def nfw(c, rd, r_bins, cosmology, delta=200, z_halo = 0):
    density = cosmology.critical_density(z_halo).to(u.Msun / u.Mpc**3).value
    md = 4/3 * np.pi * density * delta * rd**3

    Anfw = np.log(1 + c) - c / (1 + c)
    rnorm = r_bins / rd
    dmdr = md * rnorm / (rd * Anfw) / (1 / c + rnorm) ** 2
    return dmdr


def get_radius_bound_idxs(halo_radius, r_bins, r_velocities, r_counts, min_count=100):
    r_infal_idx = get_infall_radius_idx(r_bins, r_velocities)
    r200_idx = np.argmax(r_bins > halo_radius)
    r_min_idx = np.argmax(np.cumsum(r_counts) > min_count)
    return (r_min_idx, max(r200_idx, r_infal_idx) + 1)


def get_infall_radius_idx(r_bins, r_velocities):
    idxs = np.where(r_velocities > 0.0)[0]
    if len(idxs) == 0:
        return len(r_velocities) - 1
    return idxs[-1]


def get_nfw_fit(r_bins, dmdr_fit, cosmology, delta=200):
    func = lambda args: nfw(args[0], args[1], r_bins, cosmology, delta) - dmdr_fit
    fit = least_squares(func, (3, 1.0))
    return fit.x


def get_gp_fit(bin_centers, dmdr, sigma_dmdr):
    bin_centers = bin_centers.reshape(-1, 1)
    dmdr = dmdr.reshape(-1, 1)
    kernel = ConstantKernel() * RBF()

    regressor = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        alpha=(sigma_dmdr / dmdr.max()),
        n_restarts_optimizer=10,
    )
    regressor = regressor.fit(bin_centers, dmdr)
    return regressor

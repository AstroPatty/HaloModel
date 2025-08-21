import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import G, c
from astropy.cosmology import Cosmology
from pydtfe import compute_surface_density
from scipy.optimize import minimize_scalar


def fibonacci_sphere(samples=1000):
    phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle in radians
    indices = np.arange(0, samples)
    y = 1 - (indices / (samples - 1)) * 2
    radius = np.sqrt(1 - y * y)
    theta = phi * indices
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    beta = np.arcsin(z)
    alpha = np.arctan2(y, x)
    return np.vstack((alpha, beta, np.zeros_like(alpha))).T


def compute_re(kappa, cosmology, z_lens, box_size):
    resolution_degrees = (
        (box_size * u.Mpc / cosmology.angular_diameter_distance(z_lens))
        * u.radian
        / kappa.shape[0]
    )
    resolution_arcseconds = resolution_degrees.to(u.arcsec)

    kappa_th = minimize_scalar(
        func_kappa_bar_above_kappa_th,
        args=(kappa,),
        bounds=(0.0, 1.0),
        method="bounded",
    ).x
    area = len(np.where(kappa > kappa_th)[0]) * resolution_arcseconds**2
    return np.sqrt(area.value / np.pi)


def lensing_critical_density(z_lens, z_source, cosmology: Cosmology):
    Dlens = cosmology.angular_diameter_distance(z_lens)
    Dsource = cosmology.angular_diameter_distance(z_source)
    Dls = cosmology.angular_diameter_distance_z1z2(z_lens, z_source)
    sigma_cr = c**2 * Dsource / 4 / np.pi / G / Dls / Dlens
    return sigma_cr.to(u.Msun / (u.Mpc) ** 2)


def surface_density_to_kappa(surface_density, z_lens, z_source, cosmology: Cosmology):
    sigma_cr = lensing_critical_density(z_lens, z_source, cosmology)
    return surface_density / sigma_cr.value


def func_kappa_bar_above_kappa_th(kappa_th_in, kappa_in):
    res = np.mean(kappa_in[np.where(kappa_in > kappa_th_in)])
    return np.abs(res - 1.0)


def compute_re_avg(
    halo_properties,
    gravity_particles,
    particle_mass,
    z_lens,
    z_source,
    cosmology,
    samples_per_halo=50,
):
    center = [
        halo_properties["fof_halo_center_x"],
        halo_properties["fof_halo_center_y"],
        halo_properties["fof_halo_center_z"],
    ]
    particle_input = np.vstack(
        [gravity_particles["x"], gravity_particles["y"], gravity_particles["z"]]
    ).T
    rotations = fibonacci_sphere(samples=samples_per_halo)
    res = np.zeros(samples_per_halo, dtype=float)
    random = np.random.randint(0, samples_per_halo)
    for i, rotation in enumerate(rotations):
        result = compute_surface_density(
            particle_input,
            center,
            256,
            0.5,
            5,
            5,
            particle_mass,
            mc_box_width=0.007,
            mc_sample_count=5,
            rotation_angle=rotation,
        )
        if i == random:
            plt.figure(figsize=(10, 10))
            plt.imshow(np.log10(result + 1e13))
            plt.savefig(f"/home/pwells/projects/HaloModel/halos/{halo_properties['fof_halo_tag']}.png")

        kappa = surface_density_to_kappa(result, z_lens, z_source, cosmology)
        re = compute_re(kappa, cosmology, z_lens, 2)
        res[i] = re
    return {"re_mean": np.average(res), "re_std": np.std(res)}

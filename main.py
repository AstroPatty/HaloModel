from pathlib import Path

import astropy.units as u
import opencosmo as oc

from halomodel.surface import re_avg

test_data = Path("/Users/patrick/code/Production/OpenCosmo/test_data/snapshot")
haloproperties = test_data / "haloproperties.hdf5"
haloparticles = test_data / "haloparticles.hdf5"


def get_particle_mass(collection):
    if collection.simulation.n_gravity is not None:  # gravity only simulation
        n_particles = collection.simulation.n_gravity
    else:
        n_particles = collection.simulation.n_dm  #

    dm_density = collection.cosmology.Odm0 * collection.cosmology.critical_density0
    dm_density = dm_density.to(u.Msun / u.Mpc**3)
    box_size_mpc = collection.simulation.box_size / collection.cosmology.h
    total_volume = (box_size_mpc * u.Mpc) ** 3
    total_dm_mass = dm_density * total_volume
    dm_mass_per_particle = total_dm_mass / (n_particles**3)

    return dm_mass_per_particle


def main():
    structures = oc.open(haloproperties, haloparticles)
    halo_min_mass = oc.col("fof_halo_mass") > 1.4 * 10**14
    halo_max_mass = oc.col("fof_halo_mass") < 1.4 * 10**14.5
    concentration_min = oc.col("sod_halo_cdelta") > 5
    particle_mass = get_particle_mass(structures)
    structures = structures.filter(halo_min_mass, halo_max_mass, concentration_min)

    results = structures.evaluate(
        re_avg,
        insert=False,
        halo_properties=[
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
            "fof_halo_mass",
        ],
        dm_particles=["x", "y", "z"],
        format="numpy",
        particle_mass=particle_mass.value,
        z_source=10.0,
        z_lens=0.5,
        cosmology=structures.cosmology,
    )
    print(results)


if __name__ == "__main__":
    main()

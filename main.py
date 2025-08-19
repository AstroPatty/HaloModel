import astropy.units as u
import opencosmo as oc

from halomodel.gp import make_gp
from halomodel.surface import compute_re_avg


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
    structures = oc.open("data/halos.hdf5")
    dx = oc.col("sod_halo_com_x") - oc.col("fof_halo_com_x")
    dy = oc.col("sod_halo_com_y") - oc.col("fof_halo_com_y")
    dz = oc.col("sod_halo_com_z") - oc.col("fof_halo_com_z")
    xoff = (dx**2 + dy**2 + dz**2) ** 0.5 / oc.col("sod_halo_radius")
    structures = structures.with_new_columns(xoff=xoff, dataset="halo_properties")

    structures = structures.evaluate(
        make_gp,
        insert=True,
        halo_profiles=[
            "sod_halo_bin_radius",
            "sod_halo_bin_count",
            "sod_halo_bin_mass",
            "sod_halo_bin_rad_vel",
        ],
        halo_properties=[
            "sod_halo_mass",
            "sod_halo_cdelta",
            "sod_halo_radius",
        ],
        cosmology=structures.cosmology,
        format="numpy",
    )
    structures = structures.evaluate(
        compute_re_avg,
        insert=True,
        halo_properties=[
            "fof_halo_tag",
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
        ],
        gravity_particles=["x", "y", "z"],
        particle_mass=get_particle_mass(structures).value,
        z_lens=0.5,
        z_source=10,
        cosmology=structures.cosmology,
        format="numpy",
        samples_per_halo=100,
    )
    oc.write("output.hdf5", structures["halo_properties"])


if __name__ == "__main__":
    main()

import astropy.units as u
import opencosmo as oc
from pathlib import Path

from halomodel.gp import make_gp
from halomodel.surface import compute_re_avg
from mpi4py import MPI

data_root = Path("/eagle/OpenCosmo/Frontier-E-GO/analysis/step_415")
properties_path = data_root / "haloproperties" / "m000p-415.haloproperties.hdf5"
particles_path = data_root / "sodbighaloparticles" / "m000p-415.sodbighaloparticles.hdf5"
profiles_path = data_root / "sodpropertybins" / "m000p-415.sodpropertybins.hdf5"
comm = MPI.COMM_WORLD


def get_particle_mass(collection):
    if collection.simulation.n_gravity is not None:  # gravity only simulation
        n_particles = collection.simulation.n_gravity
    else:
        n_particles = collection.simulation.n_dm  #

    dm_density = collection.cosmology.Om0 * collection.cosmology.critical_density0
    dm_density = dm_density.to(u.Msun / u.Mpc**3)
    box_size_mpc = collection.simulation.box_size / collection.cosmology.h
    total_volume = (box_size_mpc * u.Mpc) ** 3
    total_dm_mass = dm_density * total_volume
    dm_mass_per_particle = total_dm_mass / (n_particles**3)

    return dm_mass_per_particle


def main():
    structures = oc.open(properties_path, particles_path, profiles_path)
    mass_low = oc.col("sod_halo_mass") > 10**14.0
    mass_high = oc.col("sod_halo_mass") < 10**14.1
    structures = structures.with_units("scalefree").filter(mass_low, mass_high).with_units("comoving")
    total = comm.gather(len(structures))
    if comm.Get_rank() == 0:
        print(f"Found {sum(total)} structures in this mass range!", flush=True)
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
            "fof_halo_tag"
        ],
        cosmology=structures.cosmology,
        format="numpy",
    )
    if comm.Get_rank() == 0:
        print("Writing GP results", flush=True)
    oc.write("/home/pwells/projects/HaloModel/halos_14_141.hdf5", structures["halo_properties"], overwrite=True)
    exit()

    cgp_min = oc.col("c_gp") > 8
    cnfw_max = oc.col("c_nfw") < 4
    structures = structures.filter(cgp_min, cnfw_max)
    print(f"Found {len(structures)} chaotic structures", flush=True)
    ds = structures["halo_properties"].filter(cgp_min, cnfw_max)
    print(len(ds), flush=True)
    if len(structures) > 10:
        structures = structures.take(10)

    if len(structures) > 0:
        structures = structures.evaluate(
            compute_re_avg,
            insert=True,
            halo_properties=[
                "fof_halo_tag",
                "fof_halo_center_x",
                "fof_halo_center_y",
                "fof_halo_center_z",
                "sod_halo_radius",
            ],
            gravity_particles=["x", "y", "z"],
            particle_mass=get_particle_mass(structures).value,
            z_lens=0.5,
            z_source=2.5,
            cosmology=structures.cosmology,
            format="numpy",
            samples_per_halo=100,
        )
    comm.Barrier()
    oc.write("/home/pwells/projects/HaloModel/structures_withre.hdf5", structures["halo_properties"])


if __name__ == "__main__":
    main()

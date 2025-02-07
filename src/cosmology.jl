
module Simulations
using Cosmology: cosmology, AbstractCosmology, H
using PhysicalConstants.CODATA2018: G
using Unitful
using UnitfulAstro



struct Simulation
    cosmology::AbstractCosmology
    boxsize::typeof(1.0UnitfulAstro.pc)
    nparticles::Int
end

function rho_crit(cosmology::AbstractCosmology)
    3 * cosmology.H0^2 / (8 * π * G)
end

function rho_crit(cosmology::AbstractCosmology, z::Number)
    hz = H(cosmology, z) # km / s / Mpc
    rho_crit = 3 * hz^2 / (8 * π * G)
    uconvert(UnitfulAstro.Msun / UnitfulAstro.Mpc^3, rho_crit)
end

function Simulation(; cosmology::AbstractCosmology, boxsize::typeof(1.0UnitfulAstro.pc), nparticles::Int)
    Simulation(cosmology, boxsize, nparticles)
end

function particle_mass(sim::Simulation)
    (sim.boxsize / sim.nparticles)^3 * sim.cosmology.Ωm * sim.cosmology.ρc / sim.cosmology.h^2
end


Planck2018 = cosmology(OmegaM=0.30966, h=0.6766)
LastJourney = Simulation(cosmology=Planck2018, boxsize=(3400 * 1e6 / Planck2018.h)UnitfulAstro.pc, nparticles=10752)

end # module Simulations








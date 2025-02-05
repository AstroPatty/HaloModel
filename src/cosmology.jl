
module Simulations
using Cosmology: cosmology, AbstractCosmology
using Unitful
using UnitfulAstro


struct Simulation
    cosmology::AbstractCosmology
    boxsize::typeof(1.0UnitfulAstro.pc)
    nparticles::Int
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








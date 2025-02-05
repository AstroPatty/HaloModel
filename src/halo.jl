module Halos

using Plots
using LsqFit

struct HaloProfile
    radius::Vector{<:Number}
    count::Vector{<:Number}
    mass::Vector{<:Number}
    vr::Vector{<:Number}
    function HaloProfile(radius::Vector{<:Number}, count::Vector{<:Number}, mass::Vector{<:Number}, vr::Vector{<:Number})
        if length(radius) != length(count) || length(radius) != length(mass) || length(radius) != length(vr)
            throw(ArgumentError("All vectors must have the same length"))
        end
        order = sortperm(radius)
        r = radius[order]
        ratio = r[2] / r[1]
        left_bin = r[1] / ratio
        r = pushfirst!(r, left_bin)
        new(r, count[order], mass[order], vr[order])
    end
end

struct DifferentialProfile
    radius::Vector{Number}
    dmdr::Vector{Number}
end

struct Halo
    properties::Dict{String,Number}
    profile::HaloProfile

end


function get_halo_property(halo::Halo, property::String)::Union{Number,Nothing}
    get(halo.properties, property, nothing)
end

function get_differential_profile(halo::Halo)::DifferentialProfile
    get_differential_profile(halo.profile)
end

function get_differential_profile(profile::HaloProfile)::DifferentialProfile
    dr = diff(profile.radius)
    dmdr = profile.mass ./ dr
    DifferentialProfile(profile.radius, dmdr)
end

function get_poisson_error(profile::HaloProfile, particle_mass::Number)::Vector{Number}
    count_in_shell = cumsum(profile.count)
    count_in_shell = pushfirst!(count_in_shell, 0)
    uncertainty = particle_mass * sqrt.(count_in_shell)
    deltam = sqrt.(uncertainty[2:end] .^ 2 .+ uncertainty[1:end-1] .^ 2)
    deltam ./ diff(profile.radius)
end

function get_poisson_error(profile::HaloProfile)::Vector{Number}
    particle_mass = profile.mass[1] / profile.count[1]
    get_poisson_error(profile, particle_mass)
end

function fit_nfw(halo::Halo, ρcrit::Number)
    # fit an NFW profile to the halo profile
    # return the best fit parameters
    profile_radius = halo.profile.radius
    bin_centers = (profile_radius[1:end-1] + profile_radius[2:end]) / 2
    fitting_function = (radius, pars) -> nfw(pars[1], pars[2], radius, ρcrit)
    fit = curve_fit(fitting_function, bin_centers, halo.profile.mass, [1.0, 1.0])
    fit.param
end

function nfw(cdelta::Number, rdelta::Number, r::AbstractVector{<:Number}, ρcrit::Number)::Vector{Number}
    delta = 200
    rho = 4 / 3 * π * (ρcrit * delta) * rdelta^3
    a = log(1 + cdelta) - cdelta / (1 + cdelta)
    rnorm = r ./ rdelta
    dmdr = rho ./ ((a .* rnorm) .* (1 / cdelta .+ rnorm) .^ 2)
    dmdr
end

function nfw(r::Number, cdelta::Number, rdelta::Number, ρcrit::Number)::Number
    mdelta = 4 / 3 * π * ρcrit * rdelta^3
    a = log(1 + cdelta) - cdelta / (1 + cdelta)
    rnorm = r / rdelta
    dmdr = mdelta / (rdelta * a) * rnorm / (1 / cdelta + rnorm)^2
    dmdr
end

function plot_differential_profile(halo::Halo)
    differential_profile = get_differential_profile(halo)
    σm = get_poisson_error(halo.profile)
    bin_centers = (differential_profile.radius[1:end-1] + differential_profile.radius[2:end]) / 2
    # plot a scatter plot with error bars
    sod_radius = get_halo_property(halo, "sod_halo_radius")
    scatter(bin_centers ./ sod_radius, differential_profile.dmdr, yerr=σm, label="dM/dR", xlabel="r/rd", ylabel="dM/dR [M⊙/h/Mpc]", title="Differential Profile", xscale=:log10, yscale=:log10)
    # log-log plot
end

end # module halo



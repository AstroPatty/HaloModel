module Halos

using Plots
using LsqFit
using Unitful

import Cosmology as CosmoBase



export Halo, HaloProfile, DifferentialProfile, get_halo_property, get_differential_profile, get_poisson_error, plot_differential_profile, cutoff_radius


struct HaloProfile{T<:AbstractFloat,U<:Integer}
    radius::Vector{T}
    count::Vector{U}
    mass::Vector{T}
    vr::Vector{T}
    function HaloProfile(radius::Vector{T}, count::Vector{U}, mass::Vector{T}, vr::Vector{T}) where {T<:AbstractFloat,U<:Integer}
        if length(radius) != length(count) || length(radius) != length(mass) || length(radius) != length(vr)
            throw(ArgumentError("All vectors must have the same length"))
        end
        order = sortperm(radius)
        r = radius[order]
        ratio = r[2] / r[1]
        left_bin = r[1] / ratio
        r = pushfirst!(r, left_bin)
        new{T,U}(r, count[order], mass[order], vr[order])
    end
end

struct DifferentialProfile{T<:AbstractFloat}
    radius::Vector{T}
    dmdr::Vector{T}
end

struct Halo{T,U}
    properties::Dict{String,Any}
    profile::HaloProfile{T,U}
end


get_halo_property(halo::Halo, property) = get(halo.properties, property, nothing)

get_differential_profile(halo::Halo) = get_differential_profile(halo.profile)

function get_differential_profile(profile::HaloProfile)::DifferentialProfile
    dr = diff(profile.radius)
    dmdr = profile.mass ./ dr
    DifferentialProfile(profile.radius, dmdr)
end

function get_poisson_error(profile::HaloProfile{T,U}, particle_mass::T) where {T,U}
    count_in_shell = cumsum(profile.count)
    count_in_shell = pushfirst!(count_in_shell, U(0))
    uncertainty = particle_mass * sqrt.(count_in_shell)
    dr = diff(profile.radius)

    result = @. (sqrt(uncertainty[2:end]^2 + uncertainty[1:end-1]^2)) / dr
    Vector{T}(result)
end

function get_poisson_error(profile::HaloProfile{T}) where {T}
    particle_mass = profile.mass[1] / T(profile.count[1])
    get_poisson_error(profile, particle_mass)
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

function cutoff_radius(halo::Halo, min_threshold::Integer)
    bin_counts = halo.profile.count
    smallest_bin_index = findfirst(cumsum(bin_counts) .> min_threshold)
    r_infall_index = findfirst(halo.profile.vr .< 0)
    r_200_index = findfirst(halo.profile.radius .> get_halo_property(halo, "sod_halo_radius"))
    return (smallest_bin_index, max(r_infall_index, r_200_index))
end

end

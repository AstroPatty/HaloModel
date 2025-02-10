module Model
using Plots
using LsqFit

using ..Halos
using ..Simulations
using Unitful
using Cosmology: FlatLCDM

export nfw, NfwProfile, fit_nfw, plot_nfw, plot_nfw!

struct NfwProfile{T<:Real,U<:Real,V<:Real,W<:Real}
    cdelta::T
    rdelta::U
    mdelta::V
    radius::Vector{W}
end

function fit_nfw(halo::Halo, with_radius_cutoffs=false)
    m200 = get_halo_property(halo, "sod_halo_mass")
    r200 = get_halo_property(halo, "sod_halo_radius")
    differential_profile = get_differential_profile(halo)
    r = differential_profile.radius
    dmdr = differential_profile.dmdr

    if with_radius_cutoffs
        smallest_bin_index, largest_bin_index = cutoff_radius(halo, 100)
        r = r[smallest_bin_index:largest_bin_index]
        dmdr = dmdr[smallest_bin_index:largest_bin_index-1]
    end

    fitting_function = (r, pars) -> nfw(r, pars[1], r200, m200)
    bin_centers = (r[1:end-1] + r[2:end]) / 2
    fit = curve_fit(fitting_function, bin_centers, dmdr, [6.0])
    NfwProfile(fit.param[1], r200, m200, bin_centers)
end


function fit_nfw(halo::Halo, cosmo::FlatLCDM, redshift, with_radius_cutoffs=false)
    # fit an NFW profile to the halo profile
    # return the best fit parameters
    h = cosmo.h
    ρcrit = Simulations.rho_crit(cosmo, redshift)
    comoving_rhocrit = ρcrit / h^2 / (1 + redshift)^3
    comoving_rhocrit = ustrip(comoving_rhocrit)

    differential_profile = get_differential_profile(halo)
    r = differential_profile.radius
    dmdr = differential_profile.dmdr

    if with_radius_cutoffs
        smallest_bin_index, largest_bin_index = cutoff_radius(halo, 100)
        r = r[smallest_bin_index:largest_bin_index]
        dmdr = dmdr[smallest_bin_index:largest_bin_index-1]
    end


    bin_centers = (r[1:end-1] + r[2:end]) / 2
    fitting_function = (radius, pars) -> nfw(radius, pars[1], pars[2]; ρcrit=comoving_rhocrit)
    fit = curve_fit(fitting_function, bin_centers, dmdr, [6.0, 1.0])
    m200 = 200 * comoving_rhocrit * 4 / 3 * π * fit.param[2]^3
    NfwProfile(fit.param[1], fit.param[2], m200, bin_centers)
end

function nfw(r::AbstractVector{T}, cdelta::Real, rdelta::Real; ρcrit::Real) where {T<:Real}
    delta = 200
    m200 = 4 / 3 * π * (ρcrit * delta) * rdelta^3
    nfw(r, cdelta, rdelta, m200)
end

function nfw(r::AbstractVector{T}, cdelta::Real, rdelta::Real, mdelta::Real) where {T<:Real}
    a = log(1 + cdelta) - cdelta / (1 + cdelta)
    rnorm = r ./ rdelta
    dmdr = @. mdelta / (rdelta * a) * rnorm / (1 / cdelta + rnorm)^2
    dmdr
end

function nfw(r::Real, cdelta::Real, rdelta::Real, ρcrit::Real)::Number
    mdelta = 4 / 3 * π * ρcrit * rdelta^3
    a = log(1 + cdelta) - cdelta / (1 + cdelta)
    rnorm = r / rdelta
    dmdr = mdelta / (rdelta * a) * rnorm / (1 / cdelta + rnorm)^2
    dmdr
end

function plot_nfw(profile::NfwProfile)
    r = profile.radius
    dmdr = nfw(r, profile.cdelta, profile.rdelta, profile.mdelta)
    plot(r, dmdr, label="NFW Profile")
end

function plot_nfw!(plot, profile::NfwProfile; label="NFW Profile")
    r = profile.radius
    dmdr = nfw(r, profile.cdelta, profile.rdelta, profile.mdelta)
    plot!(plot, r, dmdr, label=label)
end

end # module Halosmodel

module Model
using Plots
using LsqFit

using ..Halos
using ..Simulations
using Unitful
using Cosmology: FlatLCDM


struct NfwProfile
    cdelta::Float64
    rdelta::Float64
    mdelta::Float64
    radius::Vector{Float64}
end

function fit_nfw(halo::Halo, with_radius_cutoffs::Bool=false)
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

    fitting_function = (r, pars) -> nfw(pars[1], r200, m200, r)
    bin_centers = (r[1:end-1] + r[2:end]) / 2
    fit = curve_fit(fitting_function, bin_centers, dmdr, [6.0])
    NfwProfile(fit.param[1], r200, m200, bin_centers)
end


function fit_nfw(halo::Halo, cosmo::FlatLCDM, redshift::Number, with_radius_cutoffs::Bool=false)
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
    fitting_function = (radius, pars) -> nfw(pars[1], pars[2], radius, comoving_rhocrit)
    fit = curve_fit(fitting_function, bin_centers, dmdr, [6.0, 1.0])
    m200 = 200 * comoving_rhocrit * 4 / 3 * π * fit.param[2]^3
    NfwProfile(fit.param[1], fit.param[2], m200, bin_centers)
end

function nfw(cdelta::Number, rdelta::Number, r::AbstractVector{<:Number}, ρcrit::Number)::Vector{Number}
    delta = 200
    m200 = 4 / 3 * π * (ρcrit * delta) * rdelta^3
    nfw(cdelta, rdelta, m200, r)

end

function nfw(cdelta::AbstractFloat, rdelta::AbstractFloat, mdelta::AbstractFloat, r::AbstractVector{<:AbstractFloat})
    a = log(1 + cdelta) - cdelta / (1 + cdelta)
    rnorm = r ./ rdelta
    dmdr = @. mdelta / (rdelta * a) * rnorm / (1 / cdelta + rnorm)^2
    dmdr
end

function nfw(r::Number, cdelta::Number, rdelta::Number, ρcrit::Number)::Number
    mdelta = 4 / 3 * π * ρcrit * rdelta^3
    a = log(1 + cdelta) - cdelta / (1 + cdelta)
    rnorm = r / rdelta
    dmdr = mdelta / (rdelta * a) * rnorm / (1 / cdelta + rnorm)^2
    dmdr
end

function plot_nfw(profile::NfwProfile)
    r = profile.radius
    dmdr = nfw(profile.cdelta, profile.rdelta, profile.mdelta, r)
    plot(r, dmdr, label="NFW Profile")
end

function plot_nfw!(plot::Plots.Plot, profile::NfwProfile; label="NFW Profile")
    r = profile.radius
    dmdr = nfw(profile.cdelta, profile.rdelta, profile.mdelta, r)
    plot!(plot, r, dmdr, label=label)
end

end # module Halos

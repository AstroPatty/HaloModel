module model
using Plots
using LsqFit
include("convert.jl")
using .Convert

function plot_profile(radius, mass)
    r, dmdr = Convert.make_bins(radius, mass)
    # log-log plot
    plot(r, dmdr, xaxis=:log, yaxis=:log)
end

function fit_profile(r::Vector{<:Number}, dmdr::Vector{<:Number})
    # fit a power law to the data
end

function nfw_profile(r::Vector{<:Number}, cdelta, rdelta, rho_c)
    delta = 200.0
    mdelta = (4 / 3) * pi * rdelta^3 * rho_c * delta
    a = log(1.0 + cdelta) - cdelta / (1.0 + cdelta)
    rs = r ./ rdelta
    dmdr = mdelta / (rdelta * a) * (rs ./ (1 / cdelta +.rs).^2)
    return dmdr
    # return the NFW profile

end

end

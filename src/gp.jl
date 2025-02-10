module GP

using ..Halos
using ..Model
using Turing
using KernelFunctions
using LinearAlgebra
using AbstractGPs
using DataFrames


struct GpTarget{T<:AbstractFloat}
        r::Vector{T}
        dmdr::Vector{T}
        σdmdr::Vector{T}
        range_r::Tuple{T,T}
        range_dmdr::Tuple{T,T}
        nfw_f::Function
end

struct GpFit{T<:AbstractFloat,U<:AbstractFloat}
        l::T
        σf::T
        target::GpTarget{U}
end

function normalize(halo::Halo{T}, nfw_f) where {T}
        r = halo.profile.radius
        bin_centers = r[1:end-1] .+ (T(0.5) .* diff(r))
        nfw = nfw_f(bin_centers)
        dmdr = get_differential_profile(halo).dmdr
        σdmdr = get_poisson_error(halo.profile)

        # Normalize the profile
        r_rescaled = (bin_centers .- bin_centers[1]) ./ (bin_centers[end] - bin_centers[1])
        dmdr_rescaled = log10.(dmdr ./ nfw)
        dmdr_normed = (dmdr_rescaled .- minimum(dmdr_rescaled)) ./ (maximum(dmdr_rescaled) - minimum(dmdr_rescaled))
        σy_normed = σdmdr ./ (log(T(10)) * dmdr)
        return GpTarget(r_rescaled, dmdr_normed, σy_normed, (halo.profile.radius[1], halo.profile.radius[end]), (minimum(dmdr_rescaled), maximum(dmdr_rescaled)), nfw_f)
end

function normalize_radius(halo::Halo, radius::Vector{T}) where {T}
        r_rescaled = (radius .- halo.profile.radius[1]) ./ (halo.profile.radius[end] - halo.profile.radius[1])
        return r_rescaled
end


function normalize(halo::Halo)
        mdelta = get_halo_property(halo, "sod_halo_mass")
        cdelta = get_halo_property(halo, "sod_halo_cdelta")
        rdelta = get_halo_property(halo, "sod_halo_radius")
        nfw_f = (radius) -> nfw(radius, cdelta, rdelta, mdelta)
        normalize(halo, nfw_f)
end

function rescale(gp_target::GpTarget{T}, r::Vector{T}, dmdr::Vector{T}) where {T}
        # re-scale the prediction to the original units
        bin_centers = gp_target.range_r[1] .+ (r .* (gp_target.range_r[end] - gp_target.range_r[1]))
        nfw = gp_target.nfw_f(bin_centers)
        dmdr_rescaled = (dmdr .* (gp_target.range_dmdr[2] - gp_target.range_dmdr[1])) .+ gp_target.range_dmdr[1]
        dmdr_final = nfw .* 10 .^ dmdr_rescaled

        bin_centers, dmdr_final
end

# If the types are not the same
#
function rescale(gp_target::GpTarget{T}, r::Vector{S}, dmdr::Vector{S}) where {T,S}
        # re-scale the prediction to the original units
        r_ = Vector{T}(r)
        dmdr_ = Vector{T}(dmdr)
        rescale(gp_target, r_, dmdr_)
end

function covariance(gp_target::GpTarget, l, σf)
        m = gp_target.r * gp_target.r'
        uncertainty = Diagonal(gp_target.σdmdr .^ 2)
        σf^2 * exp.(-0.5 * (m .- m')^2 / l^2) + uncertainty
end


function log_prob(gp_target::GpTarget, params)
        l, σf = params
        kernel_matrix = covariance(gp_target, l, σf) + Diagonal(gp_target.σdmdr .^ 2)
        determinant = det(kernel_matrix)
        if determinant == 0
                return -Inf
        end
        inverse = inv(kernel_matrix)
        -0.5 * (gp_target.dmdr' * inverse * gp_target.dmdr + log(determinant) + length(gp_target.r) * log(2π))
end

@model function gp_model(gp_target::GpTarget)
        l ~ Uniform(0.01, 1.0)
        σf ~ Uniform(0.01, 2.0)
        log_prob(gp_target, [l, σf])
end

function gp_model(halo::Halo)
        gp_target = normalize(halo)
        chain = sample(gp_model(gp_target), MH(), MCMCThreads(), 50, 10)
        l = mean(chain[:l])
        σf = mean(chain[:σf])
        GpFit(l, σf, gp_target)
end

function predict(halo::Halo, fit::GpFit, r_pred::Vector{T}) where {T}
        r_normed = normalize_radius(halo, r_pred)
        k = fit.σf^2 * with_lengthscale(SqExponentialKernel(), fit.l)
        K_pred = kernelmatrix(k, fit.target.r, r_normed)
        Kx_pred = kernelmatrix(k, r_normed, r_normed)
        prediction = Vector{T}(Kx_pred * (K_pred \ fit.target.dmdr))
        rescale(fit.target, r_normed, prediction)
end


end

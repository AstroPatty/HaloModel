
module Convert
function make_bins(radius::Vector{<:Number}, mass::Vector{<:Number})
        # get the correct order 
        r, m = sort(radius), mass[sortperm(radius)]
        ratio = r[2] / r[1]
        left_bin = r[1] / ratio
        r = pushfirst!(r, left_bin)
        radii = 0.5 * (r[1:end-1] + r[2:end])
        dr = diff(r)
        dmdr = m ./ dr
        radii, dmdr
end

function deltam(bin_counts::Vector{<:Number}, particle_mass::Number)
        count_by_radius = prepend!(0, cumsum(bin_counts))
        # For every pair in the vector, compute the difference
        map((a, b) -> particle_mass * sqrt(a^2 + b^2), count_by_radius[1:end-1], count_by_radius[2:end])
end

end

using HDF5
using DataFrames

import .Halos


function read_properties(filename::String, group::String; n=-1)
        fid = h5open(filename, "r")
        indices = read(fid[group]["properties"], "tree_node_index")
        if n == -1
                n = length(indices)
        end
        properties = Dict{String,Vector{Any}}()
        for key in keys(fid[group]["properties"])
                properties[key] = read(fid[group]["properties"][key])[1:n]
        end
        close(fid)
        # parse into dictionary with tree_node_index as key
        output = Dict{Int64,Dict{String,Any}}()
        for (i, tni) in enumerate(indices[1:n])
                output[tni] = Dict{String,Any}()
                for key in keys(properties)
                        output[tni][key] = properties[key][i]
                end
        end
        output

end

function build_halos(filename::String, properties::Dict{Int64,Dict{String,Any}}; n=-1)::Vector{Halos.Halo}
        fid = h5open(filename, "r")
        halos = Vector{Halos.Halo}(undef, length(properties))
        tnis = collect(keys(properties))
        profile_indices = read(fid["profiles"]["offsets"]["tni"])
        index_map = findall(tni -> tni in tnis, profile_indices)

        starts = read(fid["profiles"]["offsets"]["offset"])[index_map]
        ends = starts .+ read(fid["profiles"]["offsets"]["size"])[index_map]

        if n == -1
                n = length(index_map)
        end
        for (i, tni) in enumerate(tnis[index_map[1:n]])
                halo_properties = properties[tni]
                radius = read(fid["profiles"]["data"]["sod_halo_bin_radius"])[starts[i]+1:ends[i]]
                # promote to float64
                radius = convert(Vector{Float64}, radius)
                counts = read(fid["profiles"]["data"]["sod_halo_bin_count"])[starts[i]+1:ends[i]]
                mass = read(fid["profiles"]["data"]["sod_halo_bin_mass"])[starts[i]+1:ends[i]]
                vr = read(fid["profiles"]["data"]["sod_halo_bin_rad_vel"])[starts[i]+1:ends[i]]
                profile = Halos.HaloProfile(radius, counts, mass, vr)
                halos[i] = Halos.Halo(halo_properties, profile)
        end
        close(fid)
        halos
end

function read_data(filename::String, group::String, n=-1)::Vector{Halos.Halo}
        properties = read_properties(filename, group, n=n)
        build_halos(filename, properties, n=n)
end


module DataUtils

using ProgressMeter
using JuLIP
using ACEfriction.mUtils: reinterpret
using StaticArrays, SparseArrays
export FrictionData, BlockDenseArray 
export save_h5fdata, load_h5fdata
using HDF5

struct FrictionData
    atoms
    friction_tensor
    friction_indices
end

function FrictionData(d::NamedTuple{(:at, :friction_tensor, :friction_indices)})
    return FrictionData(d.at, d.friction_tensor, d.friction_indices)
end
# function FrictionData(d::NamedTuple{(:at, :friction_tensor, :friction_indices,:friction_indices_ref)})
#     return FrictionData(d.at, d.friction_tensor, d.friction_indices, d.friction_tensor_ref)
# end

function _array2svector(x::Array{T,2}) where {T}
    return [ SVector{3}(x[i,:]) for i in 1:size(x)[1] ]
end
function _svector2array(c_vec::Vector{SVector{N_rep, T}}) where {N_rep,T<:Number}
    """
    input: c_vec each 
    N_basis = length(c_vec)
    N_rep = numbero of channels
    """
    c_matrix = Array{T}(undef,length(c_vec),N_rep)
    for j in eachindex(c_vec)
        c_matrix[j,:] = c_vec[j]
    end
    return c_matrix
end


"""
    save_h5fdata(rdata::Vector{FrictionData}, filename::String )

Saves a friction tensor data in a costum formatted hdf5 file.

### Arguments
- `rdata` : Vector{FrictionData} :
    A vector of friction data entries. Each entry is a structure of type `Frictiondata` with the following fields:
    - `at` : JuLIP.Atoms : Atoms object containing the atomic positions, cell, and periodic boundary conditions.
    - `friction_tensor` : SparseMatrix{SMatrix{3,3,Float64,9}} : Sparse matrix representation of the friction tensor.
    - `friction_indices` : Vector{Int} : Indices of the atoms for which the friction tensor is defined.
- `filename` : String : Name of the file to save to (including h5 extension).
"""
function save_h5fdata(rdata::Vector{FrictionData}, filename::String )
    fid = h5open(filename, "w")
    try
        # iterate over each data entry
        write_attribute(fid, "N_data", length(rdata))
        @showprogress for (i,d) in enumerate(rdata)
            g = create_group(fid, "$i")
            # write atoms data
            ag = create_group(g, "atoms")
            dset_pos = create_dataset(ag, "positions", Float64, (length(d.atoms.X), 3))
            for (k,x) in enumerate(d.atoms.X)
                dset_pos[k,:] = x
            end
            write_attribute(dset_pos, "column_major", true)
            write(ag, "atypes", Int.(d.atoms.Z))
            write(ag, "cell", Matrix(d.atoms.cell))
            write_attribute(ag["cell"], "column_major", true)
            write(ag, "pbc", Array(d.atoms.pbc))
            # write friction data
            fg = create_group(g, "friction_tensor")
            (I,J,V) = findnz(d.friction_tensor)
            write(fg, "ft_I", I)
            write(fg, "ft_J", J)
            dset_ft = create_dataset(fg, "ft_val", Float64, (length(V), 3, 3))
            for (k,v) in enumerate(V)
                dset_ft[k,:,:] = v
            end
            write_attribute(dset_ft, "column_major", true)
            write(fg, "ft_mask", d.friction_indices)
        end
    catch 
        close(fid)
        rethrow(e)
    end
    HDF5.close(fid)
end

function _hdf52Atoms( ag::HDF5.Group ) 
    local positions, cell
    try
        if Bool(read_attribute(ag["positions"],"column_major")) == true
            positions = read(ag["positions"])
        else
            positions = permutedims(read(ag["positions"]), [2, 1])
        end
    catch
        @warn "The attribute 'column_major' is missing for the data 'positions'. Proceed assuming array was stored in column-major format. If you are saving your array from Python, make sure to set the column_major attribute to 0 (False)."
        positions = read(ag["positions"])
    end

    try
        if Bool(read_attribute(ag["cell"],"column_major")) == true
            cell = read(ag["cell"])
        else
            cell = permutedims(read(ag["cell"]), [2, 1])
        end
    catch
        @warn "The attribute 'column_major' is missing for the data 'cell'. Proceed assuming array was stored in column-major format. If you are saving your array from Python, make sure to set the column_major attribute to 0 (False)."
        cell = read(ag["cell"])
    end
    return JuLIP.Atoms(;
                X=[SVector{3}(d) for d in eachslice(positions; dims=1)],
                Z=read(ag["atypes"]), 
                cell= cell,
                pbc=Bool.(read(ag["pbc"]))
            )
end
        
function _hdf52ft( ftg::HDF5.Group ) 
    local ft_val 
    try
        if Bool(read_attribute(ftg["ft_val"],"column_major")) == true
            ft_val = read(ftg["ft_val"])
        else
            ft_val = permutedims(read(ftg["ft_val"]), [3, 2, 1])
        end
    catch
        @warn "The attribute 'column_major' is missing for the data 'ft_val'. Proceed assuming array was stored in column-major format. If you are saving your array from Python, make sure to set the column_major attribute to 0 (False)."
        ft_val = read(ftg["ft_val"])
    end
    spft = sparse( read(ftg["ft_I"]),read(ftg["ft_J"]), [SMatrix{3,3}(d) for d in eachslice(ft_val; dims=1)] )
    ft_mask = read(ftg["ft_mask"])
    return (friction_tensor = spft, mask = ft_mask)
end


"""
    load_h5fdata(filename::String)

Loads a friction tensor data from a costum formatted hdf5 file.

### Arguments
- `filename` : String : Name of the file to load from (including h5 extension).

### Returns
- `rdata` : Vector{FrictionData} :
    A vector of friction data entries. Each entry is a structure of type `Frictiondata` with the following fields:
    - `at` : JuLIP.Atoms : Atoms object containing the atomic positions, cell, and periodic boundary conditions.
    - `friction_tensor` : SparseMatrix{SMatrix{3,3,Float64,9}} : Sparse matrix representation of the friction tensor.
    - `friction_indices` : Vector{Int} : Indices of the atoms for which the friction tensor is defined.
"""
function load_h5fdata(filename::String)
    fid = h5open(filename, "r")
    N_data = read_attribute(fid, "N_data")
    rdata = @showprogress [begin
                at = _hdf52Atoms( fid["$i/atoms/"]) 
                spft, ft_mask = _hdf52ft( fid["$i/friction_tensor/"])
                (at=at, friction_tensor=spft, friction_indices=ft_mask)
            end
            for i=1:N_data]
    HDF5.close(fid)
    return FrictionData.(rdata)
end

"""
Semi-sparse matrix representation of a square matrix M of unspecified dimension. Entries of M are 
specified by the fields `values::Matrix{T}` and `indexmap` assuming

    1. `M[k,l] = 0` if either of the indices `k`, `l` is not contained in `indexmap`.
    2. values[i,j] = M[indexmap[i],indexmap[j]]

"""
struct BlockDenseMatrix{T} <: AbstractArray{Float64,2}
    tensor::Matrix{T}
    indices
end

function BlockDenseArray(full_tensor::Matrix; indices=1:size(full_tensor,1)) 
    @assert size(full_tensor,1) == size(full_tensor,2)
    return BlockDenseMatrix(full_tensor[indices,indices], indices)
end




end

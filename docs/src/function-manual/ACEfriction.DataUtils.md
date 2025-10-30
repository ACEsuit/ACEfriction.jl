```@meta
CurrentModule = ACEfriction.DataUtils
```

The submodule `ACEfriction.DataUtils.jl` provides structures to internally store friction data in `ACEfriction.jl` and functions to import and export friction data from and to custom formatted hdf5 files.


### [Friction Data representation](@id friction-data-representation)

For pre-training storage and manipulation of friction data,
`ACEfriction.jl` implements the structure `FrictionData` to represent a single observation $(R_i,\Gamma_i)$ of an atomic configuration $R_i$ and corresponding friction tensor $\Gamma_i$: 
```julia
struct FrictionData
    atoms
    friction_tensor
    friction_indices
end
```
where
- `atoms` -- stores data of the atomic configuration ans is assumed to be of type `JuLIP.Atoms`,
- `friction_tensor` -- stores data on the friction tensor and is assumed to be of type`SparseMatrix{SMatrix{3, 3, T, 9}}`, where `T<:Float` and `Ti<:Int`. That is, the friction tensor is stored in the form of sparse matrices with $3 \times 3$-matrix valued block entries,
- `friction_indices` --  is a one-dimensional integer array, which contains all atom indices for which the friction tensor is defined.

### Importing & Exporting Friction Data

`ACEfriction.DataUtils` implements the function [`save_h5fdata`](@ref) to save arrays of friction data to a [custom  formatted](@ref costum-hdf5-format) `hdf5` file, as well as the function [`load_h5fdata`](@ref) to load friction data from such costum formatted `hdf5` files:

```@docs
save_h5fdata
```

```@docs
load_h5fdata
```


### [Custom HDF5 File Format for Friction Data](@id costum-hdf5-format)

The hierachical structure of such hdf5 files is as follows:

Each observation $(R_i,\Gamma_i)$ is saved in a separate group in named by respective index $i$, i.e., we have the following groups on root level of the hdf5 file:
```
├─ 📂 1   
├─ 📂 2  
├─ 📂 3   
│  :   
├─ 📂 N_obs   
```
Within each of these groups, the data of the respective  atomic configuration $R_i$ and friction tensor $\Gamma_i$ are saved in the subgroups `atoms` and `friction_tensor`, respectively: 
```
├─ 📂 i                        # Index of data point 
   ├─ 📂 atoms                 # Atom configuration data
   │  ├─ 🔢 atypes
   │  ├─ 🔢 cell 
   │  │  └─ 🏷️ column_major 
   │  ├─ 🔢 pbc
   │  └─ 🔢 positions
   │     └─ 🏷️ column_major
   └─ 📂 friction_tensor       # Friction tensor data
      ├─ 🔢 ft_I               
      ├─ 🔢 ft_J
      ├─ 🔢 ft_mask
      └─ 🔢 ft_val, 
         └─ 🏷️ column_major
```
 
[Datasets](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html) in the group `atoms` store the equivalent information provided by the attributes  `positions`, `numbers`, `cell`, and `pbc` of [atoms objects](https://wiki.fysik.dtu.dk/ase/ase/atoms.html), i.e., 
an atomic configuration of N atoms is described by the following datasets contained in the group `atoms`:
- `atypes` -- A one-dimensional Integer dataset of length N. The ith entry corresponds to the atomic element number of the ith atom in the configuration. (Note: `types` corresponds to the atoms attribute `numbers` in the ase.)
- `cell` -- A two-dimensional Float64 dataset of size 3 x 3. 
- `pbc` -- A one-dimensional Integer array of length 3 indicating the periodicity properties of the xyz dimension, e.g., `pbc = [1,0,0]` describes periodic boundary conditions in x dimension and non-periodic boundary conditions in the y and z dimensions. 
- `positions` -- A two-dimensional Float64 dataset of size n N x 3. The ith column corresponds to the position of the ith atom in the configuration 

[Datasets](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html) in the group `friction_tensor` store the friction tensor as a N x N sparse matrix with 3x3 valued block entries,  i.e., a friction tensor with m non-zero 3x3 blocks is stored
- `ft_I` -- A one-dimensional Integer dataset of length m specifying the column indices of non-zero block entries of the friction tensor.
- `ft_J` -- A one-dimensional Integer dataset of length m specifying the row indices of non-zero block entries of the friction tensor.
- `ft_val` -- A three-dimensional Float64 dataset of size m x 3 x 3 specifying the values of the non-zero 3 x 3 block entries of  the friction tensor. For example, the 3 x 3 array `ft_val[k,:,:]` corresponds to the 3 x 3 block entry of the friction tensor with column index `ft_I[k]`, and row index `ft_J[k]`. 
- `ft_mask` -- A one-dimensional Integer dataset/list containg the indices of atoms for which friction information is provided. 


!!! warning
    All two or three-dimensional datasets in the groups `atoms` and `friction_tensor` have an additional attribute `column_major`. If the hdf5 file is created in a language that stores matrices in column-major form (e.g., julia), this attribute must be set to 1 (True). If the hdf5 file is created in a language that stores matrices in column-major form (e.g., python), this attribute must be set to 0 (False).




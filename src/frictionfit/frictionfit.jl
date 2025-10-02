module FrictionFit

export params, FluxFrictionModel, l2_loss, weighted_l2_loss, set_params!
export flux_assemble

using ACEfriction.FrictionModels: FrictionModel
import ACEfriction.FrictionModels: get_ids
using Flux
import ACEfrictionCore: params, set_params!
using Random: randn!
using Tullio
using KernelAbstractions, CUDA
# using KernelAbstractions, CUDAKernels, CUDA
using ACEfriction.MatrixModels
using LinearAlgebra: Diagonal
include("./fluxmodels.jl")

using ACEfriction.DataUtils: FrictionData
using ACEfriction.MatrixModels: basis
using ProgressMeter
using SparseArrays
include("./fdatautils.jl")

using LinearAlgebra: I, transpose, UniformScaling, Diagonal
include("./paramtransforms.jl")


end
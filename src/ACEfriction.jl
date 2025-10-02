module ACEfriction

# utility functions for conversion of arrays, manipulation of bases and generation of bases for bond environments
include("./utils/utils.jl")
# utility functions for importing and internally storing data of friction tensors/matrices 
include("./datautils.jl")

include("./atomcutoffs.jl")
include("./matrixmodels/matrixmodels.jl")
include("./frictionmodels.jl")
include("./frictionfit/frictionfit.jl")
include("./matrixmodelsutils.jl")

import ACEfriction.FrictionModels: FrictionModel, Gamma, Sigma
export Gamma, Sigma, FrictionModel

import ACEfriction.FrictionFit: FrictionData, FluxFrictionModel, flux_assemble
export FrictionData, FluxFrictionModel, flux_assemble

import ACEfriction.MatrixModels: RWCMatrixModel, mbdpd_matrixmodel, OnsiteOnlyMatrixModel, PWCMatrixModel
export RWCMatrixModel, mbdpd_matrixmodel, OnsiteOnlyMatrixModel, PWCMatrixModel

import ACEfriction.DataUtils: write_dict, read_dict, load_h5fdata, save_h5fdata
export write_dict, read_dict, load_h5fdata, save_h5fdata

import JuLIP: Atoms
export Atoms

import ACE.ACE.ACEbonds: EllipsoidCutoff
export EllipsoidCutoff, SphericalCutoff

import ACE: Invariant, EuclideanVector, EuclideanMatrix
export Invariant, EuclideanVector, EuclideanMatrix

import JuLIP: save_dict, load_dict
export save_dict, load_dict

import ACEfriction.FrictionFit: weighted_l2_loss, weighted_l1_loss
export weighted_l2_loss, weighted_l1_loss
end

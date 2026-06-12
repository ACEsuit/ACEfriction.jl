module ACEfriction

# EquivariantTensors-based ACE backend (replaces ACEfrictionCore). Included first
# so the cutoffs / matrix models / constructors can build on it.
include("./etbackend/etbackend.jl")

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

import ACEfriction.FrictionModels: params, nparams, set_params!, scaling, get_ids, basis, matrix, randf
export params, nparams, set_params!, scaling, get_ids, basis, matrix, randf

import ACEfriction.FrictionFit: FrictionData, FluxFrictionModel, flux_assemble
export FrictionData, FluxFrictionModel, flux_assemble

import ACEfriction.MatrixModels: CWCMatrixModel, RWCMatrixModel, OnsiteOnlyMatrixModel, PWCMatrixModel
export CWCMatrixModel, RWCMatrixModel, OnsiteOnlyMatrixModel, PWCMatrixModel

import ACEfriction.DataUtils: write_dict, read_dict, load_h5fdata, save_h5fdata
export write_dict, read_dict, load_h5fdata, save_h5fdata

# Re-export AtomsBase / AtomsBuilder so existing scripts can build configurations
import AtomsBase: FlexibleSystem, FastSystem
export FlexibleSystem, FastSystem
import AtomsBuilder: bulk, rattle!
export bulk, rattle!

# cutoffs + equivariant property markers now come from the ET backend
import ACEfriction.ETBackend: EllipsoidCutoff, SphericalCutoff, SnowManCutoff
export EllipsoidCutoff, SphericalCutoff, SnowManCutoff

import ACEfriction.ETBackend: Invariant, EuclideanVector, EuclideanMatrix, SymmetricEuclideanMatrix
export Invariant, EuclideanVector, EuclideanMatrix, SymmetricEuclideanMatrix

import ACEbase.FIO: save_dict, load_dict
export save_dict, load_dict

import ACEfriction.FrictionFit: weighted_l2_loss, weighted_l1_loss
export weighted_l2_loss, weighted_l1_loss
end

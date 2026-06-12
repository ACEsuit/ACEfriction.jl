# EquivariantTensors-based ACE backend for ACEfriction.
#
# This module provides the basis/evaluation surface that ACEfriction's matrix
# models need, built on EquivariantTensors.jl (+ vendored ACEpotentials radial
# bases) instead of ACEfrictionCore.jl. It is being introduced incrementally;
# see /Users/msachs2/.claude/plans/can-you-please-help-snappy-grove.md and the
# `et-backend-migration` project memory for the migration plan.
#
# NOTE: not yet wired into the top-level `ACEfriction` module — it is developed
# and tested standalone so the package keeps compiling on the old backend until
# cut-over.

module ETBackend

# species <-> atomic number helpers (AtomsBase based)
include("species.jl")

# multi-L ET outputs -> Cartesian friction blocks (scalar/vector/matrix)
include("output_transforms.jl")

# fixed species-channel radial embedding (transform + envelope vendored from
# ACEpotentials; species folded into the radial channel)              [Phase 0]
include("radial.jl")

# thin ETACE-style onsite site basis container                    [Phase 1]
include("sitebasis.jl")

# flattened site models (basis + coefficients, no LinearACEModel)  [Phase 2]
include("models.jl")

# system-level onsite-only friction model over AtomsBase systems     [Phase 2]
include("friction_models.jl")

# native ET bond basis + Z2                                        [Phase 4]
include("bondbasis.jl")

# bond geometry: ellipsoid cutoff, env transform, bond iterator    [Phase 6]
include("bond_env.jl")

# flattened offsite model + system-level PWC friction model        [Phase 6]
include("offsite_models.jl")

# write_dict / read_dict (recipe-based) for basis + models         [Phase 5]
include("serialization.jl")

end # module

module AtomCutoffs

# Cutoffs for the ET backend. SphericalCutoff (onsite + atom-centred offsite) and
# EllipsoidCutoff (bond-centred offsite) come from ETBackend; this module just
# re-exports them and the environment transforms so the matrix models can use a
# single name. (Replaces the ACEfrictionCore/ACEbonds cutoffs.)

import ACEfriction.ETBackend
import ACEfriction.ETBackend: SphericalCutoff, EllipsoidCutoff, SnowManCutoff, env_cutoff,
                              env_filter, ellipsoid_env_transform,
                              spherical_bond_transform, et_bonds
import ACEfriction.ETBackend: _atomic_number, _chemical_symbol

export SphericalCutoff, EllipsoidCutoff, SnowManCutoff, AbstractCutoff
export env_filter, env_cutoff
export ellipsoid_env_transform, spherical_bond_transform, et_bonds

const AbstractCutoff = Union{SphericalCutoff, EllipsoidCutoff, SnowManCutoff}

end

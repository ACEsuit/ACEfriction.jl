# Species conversions for the ET backend (AtomsBase-based, no ACEfrictionCore).
# Species are represented internally as plain `Int` atomic numbers (the AtomsBase
# / ACEpotentials convention). These helpers convert between `Int` atomic numbers
# and element `Symbol`s (e.g. :Cu) for basis labels and user-facing input.
#
# Ported verbatim from ACEfrictionCore/src/utils/species.jl so the ET backend
# carries no dependency on ACEfrictionCore.

import AtomsBase
using AtomsBase: ChemicalSpecies

# Int atomic number -> element Symbol (e.g. 29 -> :Cu)
_chemical_symbol(z::Integer) = Symbol(ChemicalSpecies(z))
_chemical_symbol(s::Symbol) = s
_chemical_symbol(s::ChemicalSpecies) = Symbol(s)

# element Symbol / ChemicalSpecies -> Int atomic number (e.g. :Cu -> 29).
# Route Symbols through ChemicalSpecies (not the bare `atomic_number(::Symbol)` dict
# lookup) so dummy/non-element species such as `:X` (Z=0) resolve.
_atomic_number(z::Integer) = Int(z)
_atomic_number(s::Symbol) = Int(AtomsBase.atomic_number(ChemicalSpecies(s)))
_atomic_number(s::ChemicalSpecies) = Int(AtomsBase.atomic_number(s))
# strings arise from JSON-serialized dict keys: "29" (atomic number) or "Cu" (symbol)
function _atomic_number(s::AbstractString)
   z = tryparse(Int, s)
   return z === nothing ? Int(AtomsBase.atomic_number(ChemicalSpecies(Symbol(s)))) : z
end

module BondSelectors

import ACE: AbstractSparseBasis, maxorder, Prodb, Onepb, OneParticleBasis, 
            degree, CategorySparseBasis, SparseBasis 
import Base: filter 

struct SparseCylindricalBondBasis <: AbstractSparseBasis
   maxorder::Int
   weight::Dict
   maxlevels::Dict
   p::Float64
   besym::Symbol
   bondsym::Symbol 
   envsym::Symbol 
   ksym::Symbol
   lsym::Symbol
   weight_cat::Dict 
end


@noinline function SparseCylindricalBondBasis(; 
               besym = :be, bondsym = :bond, envsym = :env, ksym = :k,  
               lsym = :l, 
               maxorder = nothing, 
               p = 1, 
               weight = Dict{Symbol, Float64}(), 
               default_maxdeg = nothing,
               maxlevels = nothing,
               weight_cat = Dict(bondsym => 1.0, envsym => 1.0), 
               )
   @show maxorder, default_maxdeg
   if (default_maxdeg != nothing) && (maxlevels == nothing)
      return SparseCylindricalBondBasis(maxorder, weight, 
                          Dict("default" => default_maxdeg), 
                          p, besym, bondsym, envsym, ksym, lsym, weight_cat)
   elseif (default_maxdeg == nothing) && (maxlevels != nothing)
      return SparseCylindricalBondBasis(maxorder, weight, maxlevels, 
                          p, besym, bondsym, envsym, ksym, lsym, weight_cat)
   else
      @error """Either both or neither optional arguments `maxlevels` and 
                `default_maxdeg` were provided. To avoid ambiguity ensure that 
                exactly one of these arguments is provided."""
   end
end


function Base.filter(b::Onepb, Bsel::SparseCylindricalBondBasis, basis::OneParticleBasis)
   d = degree(b, basis)
   k = b[Bsel.ksym]
   return (k == 1 && b[Bsel.besym] == Bsel.envsym) || 
            (d == k-1 && b[Bsel.besym] == Bsel.bondsym) 
end

function Base.filter(bb::Prodb, Bsel::SparseCylindricalBondBasis, basis::OneParticleBasis)
   if length(bb) == 0; return true; end 
   has1bond = count((b[Bsel.besym] == Bsel.bondsym) for b in bb) == 1
   isinvariant = sum( b[Bsel.lsym] for b in bb ) == 0 
   return has1bond && isinvariant
end

# maxorder and maxlevel are inherited from the abstract interface 

level(b::Union{Prodb, Onepb}, Bsel::SparseCylindricalBondBasis, basis::OneParticleBasis) =
      cat_weighted_degree(b, Bsel, basis)


# Category-weighted degree function
cat_weighted_degree(b::Onepb, Bsel::SparseCylindricalBondBasis, basis::OneParticleBasis) =
      degree(b, basis, Bsel.weight) * Bsel.weight_cat[getproperty(b, Bsel.isym)]

cat_weighted_degree(bb::Prodb, Bsel::SparseCylindricalBondBasis, basis::OneParticleBasis) = (
      length(bb) == 0 ? 0.0
                      : norm(cat_weighted_degree.(bb, Ref(Bsel), Ref(basis)), Bsel.p)
      )

"""
Constructors of this alias of ACE.CategorySparseBasis can be used to conveniently create 
basis selectors for ACE bond bases that are defined on ellipsoid-shaped bond environments 
implmeneted as `EllipsoidCutoff` in the sub-module `ACEbonds.BondCutoffs``. 
"""
const EllipsoidBondBasis = CategorySparseBasis

function EllipsoidBondBasis(Bsel::SparseBasis; 
   isym=:mube, bond_weight = 1.0,  species =[:X], 
   species_minorder_dict = Dict{Symbol,Int64}(), species_maxorder_dict = Dict{Symbol,Int64}(), 
   species_weight_cat =  Dict( s => 1.0 for s in species)) 
   return CategorySparseBasis(isym, cat([:bond],species,dims=1);
      maxorder = maxorder(Bsel), 
      p = Bsel.p, 
      weight = Bsel.weight, 
      maxlevels = Bsel.maxlevels,
      minorder_dict = merge(Dict( :bond => 1), species_minorder_dict),
      maxorder_dict = merge(Dict( :bond => 1), species_maxorder_dict),
      weight_cat = merge(Dict(:bond => bond_weight), species_weight_cat) 
   )
end


function EllipsoidBondBasis(;  
      maxorder::Integer = nothing, 
      p = 1, 
      weight = Dict(:l => 1.0, :n => 1.0), 
      default_maxdeg = nothing,
      maxlevels::Dict{Any, Float64} = nothing,
      isym=:mube, bond_weight = 1.0,  species =[:X],
      species_minorder_dict = Dict{Any, Float64}(),
      species_maxorder_dict = Dict{Any, Float64}(),
      species_weight_cat = Dict(c => 1.0 for c in species), 
   ) 
   Bsel = SparseBasis(;  maxorder = maxorder, 
                        p = p, 
                        weight = weight, 
                        default_maxdeg = default_maxdeg, 
                        maxlevels = maxlevels ) 
   return EllipsoidBondBasis(Bsel; 
      isym=isym, bond_weight = bond_weight,  species = species, 
   species_minorder_dict = species_minorder_dict, species_maxorder_dict = species_maxorder_dict, 
   species_weight_cat = species_weight_cat) 
end

    
end
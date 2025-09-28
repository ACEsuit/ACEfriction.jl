using ACEbonds.BondSelectors: EllipsoidBondBasis
using ACE

export SymmetricEllipsoidBondBasis

  # explicitly included all optional arguments for transparancy
 function SymmetricEllipsoidBondBasis(ϕ::ACE.AbstractProperty; 
       maxorder::Integer = nothing, 
       p = 1, 
       weight = Dict(:l => 1.0, :n => 1.0), 
       default_maxdeg = nothing,
       #maxlevels::Dict{Any, Float64} = nothing,
       r0 = .4, 
       rin=.0, 
       trans = PolyTransform(2, r0), 
       pcut=2, 
       pin=2, 
       bondsymmetry=nothing,
       kvargs...) # kvargs = additional optional arguments for EllipsoidBondBasis: i.e., species =[:X], isym=:mube, bond_weight = 1.0,  species_minorder_dict = Dict{Any, Float64}(), species_maxorder_dict = Dict{Any, Float64}(), species_weight_cat = Dict(c => 1.0 for c in species), 
       Bsel = SparseBasis(;  maxorder = maxorder, 
                         p = p, 
                         weight = weight, 
                         default_maxdeg = default_maxdeg)
                         #maxlevels = maxlevels ) 
       return SymmetricEllipsoidBondBasis(ϕ, Bsel; r0=r0, rin=rin,trans=trans, pcut=pcut, pin=pin,bondsymmetry=bondsymmetry, kvargs...)                 
 end

 
 function SymmetricEllipsoidBondBasis(ϕ::ACE.AbstractProperty, Bsel::ACE.SparseBasis; 
      r0 = .4, 
      rin=.0, 
      trans = polytransform(2, r0), 
      pcut=2, 
      pin=2, 
      bondsymmetry=nothing, 
      species =[:X], 
      kvargs...
   )
   if haskey(kvargs,:isym) 
      @assert kvargs[:isym] == :mube
   end
   @assert 0.0 < r0 < 1.0
   @assert 0.0 <= rin < 1.0

   BondSelector = EllipsoidBondBasis( Bsel; species=species, kvargs...)
   min_weight = minimum(values(BondSelector.weight_cat))
   maxdeg = Int(ceil(maximum(values(BondSelector.maxlevels))))
   RnYlm = ACE.Utils.RnYlm_1pbasis(;  r0 = r0, 
      rin = rin,
      trans = trans, 
      pcut = pcut,
      pin = pin, 
      rcut= 1.0,
      Bsel = Bsel,
      maxdeg= maxdeg * max(1,Int(ceil(1/min_weight)))
   );
   Bc = ACE.Categorical1pBasis(cat([:bond],species, dims=1); varsym = :mube, idxsym = :mube )
   B1p =  Bc * RnYlm 
   return SymmetricEllipsoidBondBasis(ϕ, BondSelector, B1p; bondsymmetry=bondsymmetry)
end

function SymmetricEllipsoidBondBasis(ϕ::ACE.AbstractProperty, BondSelector::EllipsoidBondBasis, B1p::ACE.Product1pBasis; bondsymmetry=nothing)
   filterfun = _->true
   if bondsymmetry == "Invariant"
      filterfun = ACE.EvenL(:mube, [:bond])
   end
   if bondsymmetry == "Covariant"
      filterfun = x -> !(ACE.EvenL(:mube, [:bond])(x))
   end
   return ACE.SymmetricBasis(ϕ, B1p, BondSelector; filterfun = filterfun)
end


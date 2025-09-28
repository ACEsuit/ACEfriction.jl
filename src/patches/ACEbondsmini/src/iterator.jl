using JuLIP, StaticArrays, LinearAlgebra
using ACE: State, filter 
using JuLIP.Potentials: neigsz
using JuLIP: Atoms
# using ACE: BondEnvelope, filter, State, CylindricalBondEnvelope
import ACEbonds.BondCutoffs: env_cutoff
using ACEbonds.BondCutoffs: AbstractBondCutoff, env_filter, EllipsoidCutoff

_msort(z1,z2) = (z1<=z2 ? (z1,z2) : (z2,z1)) #TODO: this is hack. Need to either not use it here or define it once across all packages.
#env_cutoff(cutoff::EllipsoidCutoff) = max(cutoff.rcutbond*.5 + cutoff.zcutenv, sqrt((cutoff.rcutbond*.5)^2+ cutoff.rcutenv^2))
env_cutoff(cutoffs::Dict{Tuple{AtomicNumber,AtomicNumber},CUTOFF}) where {CUTOFF<:AbstractBondCutoff} = maximum(env_cutoff(c) for c in values(cutoffs))


bonds(at::Atoms, env::AbstractBondCutoff, args...) = 
         bonds( at, env.rcutbond, env_cutoff(env), 
                       (r, z) -> env_filter(r, z, env), args...)



# function bonds(at::Atoms, rcutbond, rcutenv, env_filter; subset=nothing) 
#    return (subset === nothing ? bonds(at::Atoms, rcutbond, rcutenv, env_filter) 
#                               : bonds(at::Atoms, rcutbond, rcutenv, env_filter,subset))
# end

# TODO: make this type-stable
struct BondsIterator 
   at
   nlist_bond
   nlist_env 
   filter 
end 

"""
* rcutbond: include all bonds (i,j) such that rij <= rcutbond 
* `rcutenv`: include all bond environment atoms k such that `|rk - mid| <= rcutenv` 
* `filter` : `filter(X) == true` if particle `X` is to be included; `false` if to be discarded from the environment
"""
function bonds(at::Atoms, rcutbond, rcutenv, filter) 
   nlist_bond = neighbourlist(at, rcutbond; recompute=true, storelist=false) 
   nlist_env = neighbourlist(at, rcutenv; recompute=true, storelist=false)
   return BondsIterator(at, nlist_bond, nlist_env, filter)
end

function Base.iterate(iter::BondsIterator, state=(1,0))
   i, q = state 
   # store temporary arrays for those...
   Js, Rs = neigs(iter.nlist_bond, i)

   # nothing left to do 
   if i >= length(iter.at) && q >= length(Js)
      return nothing 
   end 

   # increment: 
   #   case 1: we haven't yet exhausted the current neighbours. 
   #           just increment the q index pointing into Js 
   if q < length(Js)
      q += 1
      # here we could build in a rule to skip any pair for which we don't 
      # want to do the computation. 

    #  case 2: if i < length(at) but q >= length(Js) then we need to 
    #          increment the i index and get the new neighbours 
   elseif i < length(iter.at) 
      i += 1 
      Js, Rs = neigs(iter.nlist_bond, i)
      q = 1 
   else 
      return nothing 
   end 

   j = Js[q]   # index of neighbour (in central cell)
   rr0 = rrij = Rs[q]  # position of neighbour (in shifted cell) relative to i
   # ssj = Rs[q] - iter.at.X[j]   # shift of atom j into shifted cell
   
   # now we construct the environment 
   Js_e, Rs_e, Zs_e = _get_bond_env(iter, i, j, rrij)

   return (i, j, rrij, Js_e, Rs_e, Zs_e), (i, q)
end


function _get_bond_env(iter::BondsIterator, i, j, rrij)
   # TODO: store temporary arrays 
   Js_i, Rs_i, Zs_i = neigsz(iter.nlist_env, iter.at, i)

   rri = iter.at.X[i]
   rrmid = rri + 0.5 * rrij
   Js = Int[]; sizehint!(Js,  length(Js_i) ÷ 4)
   Rs = typeof(rrij)[]; sizehint!(Rs,  length(Js_i) ÷ 4)
   Zs = AtomicNumber[]; sizehint!(Zs,  length(Js_i) ÷ 4)

   ŝ = rrij/norm(rrij) 
   
   # find the bond and remember it; 
   # TODO: this could now be integrated into the second loop 
   q_bond = 0 
   for (q, rrq) in enumerate(Rs_i)
      # rr = rrq + rri - rrmid 
      if rrq ≈ rrij   # TODO: replace this with checking for j and shift!
         @assert Js_i[q] == j
         q_bond = q 
         break 
      end
   end
   if q_bond == 0 
      error("the central bond neigbour atom j was not found")
   end

   # now add the environment 
   for (q, rrq) in enumerate(Rs_i)
      # skip the central bond 
      if q == q_bond; continue; end 
      # add the rest provided they fall within the provided filter 
      rr = rrq + rri - rrmid 
      z = dot(rr, ŝ)
      r = norm(rr - z * ŝ)
      if iter.filter(r, z)
         push!(Js, Js_i[q])
         push!(Rs, rr)
         push!(Zs, Zs_i[q])
      end
   end

   return Js, Rs, Zs 
end


# using ACE: BondEnvelope, filter, State, CylindricalBondEnvelope

# TODO: make this type-stable




struct FilteredBondsIterator
   at
   nlist_bond
   nlist_env 
   env_filter
   subset
end 

"""
* rcutbond: include all bonds (i,j) such that rij <= rcutbond 
* `rcutenv`: include all bond environment atoms k such that `|rk - mid| <= rcutenv` 
* `env_filter` : `env_filter(X) == true` if particle `X` is to be included; `false` if to be discarded from the environment
* `subset` : can either be of type Array{<:Int} in which case the bond iterator iterates only over  bonds between atom pairs where the indices of both atoms are contained in indsf. 
Alternatively, indsf can also be of the form of a filter function `atom_filter(i::Int,at::AbstractAtoms)::Bool`, that returns `true` if bonds to the ith atom
   in the configuration `at` are to be included in the iterator, and `false`` otherwise. Consequently, the iterator only iterates over bonds between atom pairs
   where both atoms satisfy the filter criterion.
"""
bonds(at::Atoms, rcutbond, rcutenv, env_filter, subset) = FilteredBondsIterator(at, rcutbond, rcutenv, env_filter, subset)
bonds(at::Atoms, cutoff::AbstractBondCutoff, filter=(_,_)->true) = FilteredBondsIterator( at, cutoff.rcutbond, 
                                                                   env_cutoff(cutoff) ,
                                                                  (r, z) -> env_filter(r, z, cutoff),  filter )

"""
* rcutbond: include all bonds (i,j) such that rij <= rcutbond 
* `rcutenv`: include all bond environment atoms k such that `|rk - mid| <= rcutenv` 
* `env_filter` : `env_filter(X) == true` if particle `X` is to be included; `false` if to be discarded from the environment
"""
function FilteredBondsIterator(at::Atoms, rcutbond::Real, rcutenv::Real, env_filter, subset::Array{<:Int}) 
   nlist_bond = neighbourlist(at, rcutbond; recompute=true, storelist=false) 
   nlist_env = neighbourlist(at, rcutenv; recompute=true, storelist=false)
   return FilteredBondsIterator(at, nlist_bond, nlist_env, env_filter, subset)
end

function FilteredBondsIterator(at::Atoms, rcutbond::Real, rcutenv::Real, env_filter, filter) 
   subset = findall(i->filter(i,at), 1:length(at) )
    #@show inds
   return FilteredBondsIterator(at, rcutbond, rcutenv, env_filter, subset) 
end


function increment(iter::FilteredBondsIterator, state)
    ic, ib, Js, Rs = state
    ib = ib + 1 # increase bond index
    if ib > length(Js) # already visited/iterated over all atoms in environment ? 
        ic = ic + 1 # increase index of center atom 
        if ic > length(iter.subset) # all relevant center atoms already visited?  
            return (nothing, ib, Js, Rs) # if yes, done! 
        else
            ib = 1 # if no start at first atom in next environment
            Js, Rs = neigs(iter.nlist_bond, iter.subset[ic])
        end
    end 
    return (ic, ib, Js, Rs)
end

function Base.iterate(iter::FilteredBondsIterator)
   # if none of the atoms satisfy the filter criterion, there is nothing to iterate over
   if length(iter.subset) == 0
      return nothing
   else
      Js, Rs = neigs(iter.nlist_bond, iter.subset[1])
      state = (1,0,Js,Rs)
      return iterate(iter, state)
   end
end

function Base.iterate(iter::FilteredBondsIterator, state)
   ic, ib, Js, Rs = state 

   # Check whether s must be incremented (jumpt to next centre atom) or nothing left to iterate over
   if ic > length(iter.subset)    # nothing left to do 
    return nothing
   end
   #println("Before while")
   #@show Js
   while(true)
        (ic, ib, Js, Rs) = increment(iter, (ic, ib, Js, Rs))
        if isnothing(ic)
            return nothing
        elseif !isempty(Js) && Js[ib] in iter.subset # here we could add a finer filter criterion, e.g. iter.fiter(iter.subset[ic], Js[ib], iter.at )
            break
        end
   end
   i = iter.subset[ic]
   j = Js[ib]   # index of neighbour (in central cell)
   rrij = Rs[ib]  # position of neighbour (in shifted cell) relative to i
   # ssj = Rs[q] - iter.at.X[j]   # shift of atom j into shifted cell
   # @show (i,j)
   # now we construct the environment 
   Js_e, Rs_e, Zs_e = _get_bond_env(iter, i, j, rrij)

   return (i, j, rrij, Js_e, Rs_e, Zs_e), (ic, ib, Js, Rs)
end


function _get_bond_env(iter::FilteredBondsIterator, i, j, rrij)
   # TODO: store temporary arrays 
   Js_i, Rs_i, Zs_i = neigsz(iter.nlist_env, iter.at, i)

   rri = iter.at.X[i]
   rrmid = rri + 0.5 * rrij
   Js = Int[]; sizehint!(Js,  length(Js_i) ÷ 4)
   Rs = typeof(rrij)[]; sizehint!(Rs,  length(Js_i) ÷ 4)
   Zs = AtomicNumber[]; sizehint!(Zs,  length(Js_i) ÷ 4)

   ŝ = rrij/norm(rrij) 
   
   # find the bond and remember it; 
   # TODO: this could now be integrated into the second loop 
   q_bond = 0 
   for (q, rrq) in enumerate(Rs_i)
      # rr = rrq + rri - rrmid 
      if rrq ≈ rrij   # TODO: replace this with checking for j and shift!
         @assert Js_i[q] == j
         q_bond = q 
         break 
      end
   end
   if q_bond == 0 
      error("the central bond neigbour atom j was not found")
   end

   # now add the environment 
   for (q, rrq) in enumerate(Rs_i)
      # skip the central bond 
      if q == q_bond; continue; end 
      # add the rest provided they fall within the provided env_filter 
      rr = rrq + rri - rrmid 
      z = dot(rr, ŝ)
      r = norm(rr - z * ŝ)
      if iter.env_filter(r, z) #TODO: by modifying the env_filter function we could allow for species pair-dependent Ellipsoid cutoffs.
         push!(Js, Js_i[q])
         push!(Rs, rr)
         push!(Zs, Zs_i[q])
      end
   end

   return Js, Rs, Zs 
end

struct FilteredBondsIteratorVarCutoff
   at
   nlist_bond
   nlist_env 
   subset
   cutoffs
end 

"""
* rcutbond: include all bonds (i,j) such that rij <= rcutbond 
* `rcutenv`: include all bond environment atoms k such that `|rk - mid| <= rcutenv` 
* `env_filter` : `env_filter(r,z,zzi,zzj) == true` if particle `X` is to be included; `false` if to be discarded from the environment
* `subset` : can either be of type Array{<:Int} in which case the bond iterator iterates only over  bonds between atom pairs where the indices of both atoms are contained in indsf. 
Alternatively, indsf can also be of the form of a filter function `atom_filter(i::Int,at::AbstractAtoms)::Bool`, that returns `true` if bonds to the ith atom
   in the configuration `at` are to be included in the iterator, and `false`` otherwise. Consequently, the iterator only iterates over bonds between atom pairs
   where both atoms satisfy the filter criterion.
"""
function bonds(at::Atoms, cutoffs::Dict{Tuple{AtomicNumber,AtomicNumber},CUTOFF}, subset::Array{<:Int}) where {CUTOFF<:AbstractBondCutoff}
   rcutbond =  maximum(cutoff.rcutbond for cutoff in values(cutoffs))
   rcutenv = env_cutoff(cutoffs)
   return FilteredBondsIteratorVarCutoff(at, rcutbond, rcutenv,subset, cutoffs)
end

function bonds(at::Atoms, cutoffs::Dict{Tuple{AtomicNumber,AtomicNumber},CUTOFF}, filter= _->true) where {CUTOFF<:AbstractBondCutoff}
   subset = findall(i->filter(i,at), 1:length(at) )
   return bonds(at, cutoffs, subset) 
end




"""
* rcutbond: include all bonds (i,j) such that rij <= rcutbond 
* `rcutenv`: include all bond environment atoms k such that `|rk - mid| <= rcutenv` 
* `env_filter` : `env_filter(X) == true` if particle `X` is to be included; `false` if to be discarded from the environment
"""
function FilteredBondsIteratorVarCutoff(at::Atoms, rcutbond::Real, rcutenv::Real, subset::Array{<:Int}, cutoffs) 
   nlist_bond = neighbourlist(at, rcutbond; recompute=true, storelist=false) 
   nlist_env = neighbourlist(at, rcutenv; recompute=true, storelist=false)
   return FilteredBondsIteratorVarCutoff(at, nlist_bond, nlist_env, subset, cutoffs)
end

function FilteredBondsIteratorVarCutoff(at::Atoms, rcutbond::Real, rcutenv::Real, env_filter, filter) 
   subset = findall(i->filter(i,at), 1:length(at) )
    #@show inds
   return FilteredBondsIteratorVarCutoff(at, rcutbond, rcutenv, env_filter, subset) 
end


function increment(iter::FilteredBondsIteratorVarCutoff, state)
    ic, ib, Js, Rs = state
    ib = ib + 1 # increase bond index
    if ib > length(Js) # already visited/iterated over all atoms in environment ? 
        ic = ic + 1 # increase index of center atom 
        if ic > length(iter.subset) # all relevant center atoms already visited?  
            return (nothing, ib, Js, Rs) # if yes, done! 
        else
            ib = 1 # if no start at first atom in next environment
            Js, Rs = neigs(iter.nlist_bond, iter.subset[ic])
        end
    end 
    return (ic, ib, Js, Rs)
end

function Base.iterate(iter::FilteredBondsIteratorVarCutoff)
   # if none of the atoms satisfy the filter criterion, there is nothing to iterate over
   if length(iter.subset) == 0
      return nothing
   else
      Js, Rs = neigs(iter.nlist_bond, iter.subset[1])
      state = (1,0,Js,Rs)
      return iterate(iter, state)
   end
end

function Base.iterate(iter::FilteredBondsIteratorVarCutoff, state)
   ic, ib, Js, Rs = state 
   Zs = iter.at.Z[Js]
   # Check whether s must be incremented (jumpt to next centre atom) or nothing left to iterate over
   if ic > length(iter.subset)    # nothing left to do 
    return nothing
   end
   #println("Before while")
   #@show Js
   while(true)
        (ic, ib, Js, Rs) = increment(iter, (ic, ib, Js, Rs))
        if isnothing(ic)
            return nothing
        elseif !isempty(Js) && Js[ib] in iter.subset && haskey(iter.cutoffs,_msort(iter.at.Z[iter.subset[ic]],iter.at.Z[Js[ib]])) && norm(Rs[ib]) < iter.cutoffs[_msort(iter.at.Z[iter.subset[ic]],iter.at.Z[Js[ib]])].rcutbond  # here we could add a finer filter criterion, e.g. iter.fiter(iter.subset[ic], Js[ib], iter.at )
            break
        end
   end
   i = iter.subset[ic]
   j = Js[ib]   # index of neighbour (in central cell)
   rrij = Rs[ib]  # position of neighbour (in shifted cell) relative to i
   # ssj = Rs[q] - iter.at.X[j]   # shift of atom j into shifted cell
   # @show (i,j)
   # now we construct the environment 
   Js_e, Rs_e, Zs_e = _get_bond_env(iter, i, j, rrij)

   return (i, j, rrij, Js_e, Rs_e, Zs_e), (ic, ib, Js, Rs)
end


function _get_bond_env(iter::FilteredBondsIteratorVarCutoff, i, j, rrij)
   # TODO: store temporary arrays 
   Js_i, Rs_i, Zs_i = neigsz(iter.nlist_env, iter.at, i)

   rri = iter.at.X[i]
   rrmid = rri + 0.5 * rrij
   Js = Int[]; sizehint!(Js,  length(Js_i) ÷ 4)
   Rs = typeof(rrij)[]; sizehint!(Rs,  length(Js_i) ÷ 4)
   Zs = AtomicNumber[]; sizehint!(Zs,  length(Js_i) ÷ 4)

   ŝ = rrij/norm(rrij) 
   
   # find the bond and remember it; 
   # TODO: this could now be integrated into the second loop 
   q_bond = 0 
   for (q, rrq) in enumerate(Rs_i)
      # rr = rrq + rri - rrmid 
      if rrq ≈ rrij   # TODO: replace this with checking for j and shift!
         @assert Js_i[q] == j
         q_bond = q 
         break 
      end
   end
   if q_bond == 0 
      error("the central bond neigbour atom j was not found")
   end

   # now add the environment 
   cutoff = iter.cutoffs[_msort(iter.at.Z[i], iter.at.Z[j])]
   for (q, rrq) in enumerate(Rs_i)
      # skip the central bond 
      if q == q_bond; continue; end 
      # add the rest provided they fall within the provided env_filter 
      rr = rrq + rri - rrmid 
      z = dot(rr, ŝ)
      r = norm(rr - z * ŝ)
      if env_filter(r, z, cutoff) #TODO: by modifying the env_filter function we could allow for species pair-dependent Ellipsoid cutoffs.
         push!(Js, Js_i[q])
         push!(Rs, rr)
         push!(Zs, Zs_i[q])
      end
   end

   return Js, Rs, Zs 
end



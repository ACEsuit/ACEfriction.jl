
using .ACEbonds.BondCutoffs: env_transform, rrule_env_transform, env_filter, AbstractBondCutoff

import ACEfriction.ACE: params, nparams, set_params!

export params, nparams, set_params!
# TODO: extend implementation to allow for LinearModels with multiple featuers. 

struct ACEBondPotential{TM} <: AbstractCalculator
   models::Dict{Tuple{AtomicNumber, AtomicNumber}, TM}
   cutoff::AbstractBondCutoff{Float64}
end


struct ACEBondPotentialBasis{TM} <: JuLIP.MLIPs.IPBasis
   models::Dict{Tuple{AtomicNumber, AtomicNumber}, TM}  # model = basis
   inds::Dict{Tuple{AtomicNumber, AtomicNumber}, UnitRange{Int}}
   cutoff::AbstractBondCutoff{Float64}
end

function basis(V::ACEBondPotential)
   models = Dict( [zz => model.basis for (zz, model) in V.models]... )
   inds = _get_basisinds(V)
   return ACEBondPotentialBasis(models, inds, V.cutoff)
end

ACEBondCalc = Union{ACEBondPotential, ACEBondPotentialBasis}

function params(calc::ACEBondPotential) 
   θ = zeros(nparams(calc))
   inds = _get_basisinds(calc)
   for zz in keys(inds)
       m = _get_model(calc, zz[1],zz[2])
       θ[inds[zz]] = params(m) 
   end
   return θ
end

function set_params!(calc::ACEBondPotential, θ)
   inds =  _get_basisinds(calc)
   for zz in keys(calc.models)
      ACE.set_params!(calc, zz, θ[inds[zz]])
   end
end

function set_params!(calc::ACEBondPotential, zz::Tuple{AtomicNumber,AtomicNumber}, θ)
   set_params!(_get_model(calc, zz[1], zz[2]),θ)
end

nparams(V::ACEBondPotential) = sum(length(inds) for (_, inds) in _get_basisinds(V))


Base.length(basis::ACEBondPotentialBasis) = 
      sum(length(inds) for (_, inds) in basis.inds)


function _get_basisinds(V::ACEBondPotential)
   inds = Dict{Tuple{AtomicNumber, AtomicNumber}, UnitRange{Int}}()
   zz = sort(collect(keys(V.models)))
   i0 = 0
   for z in zz
      mo = V.models[z]
      len = length(mo.basis)
      inds[z] = (i0+1):(i0+len)   # to generalize for general models
      i0 += len
   end
   return inds 
end

_get_basisinds(V::ACEBondPotentialBasis) = V.inds

# --------------------------------------------------------

import JuLIP: energy
#, forces, virial 
import ACEfriction.ACE: evaluate #, evaluate_d, grad_config


# overload the initiation of the bonds iterator to correctly extract the 
# right cutoffs. 
bonds(at::Atoms, calc::ACEBondCalc, args...) = bonds(at, calc.cutoff, args...) 


_get_model(calc::ACEBondCalc, zi, zj) = 
      calc.models[(min(zi, zj), max(zi,zj))]

function energy(calc::ACEBondPotential, at::Atoms)
   E = 0.0 
   for (i, j, rrij, Js, Rs, Zs) in bonds(at, calc)
      # find the right ace model 
      ace = _get_model(calc, at.Z[i], at.Z[j])
      # transform the euclidean to cylindrical coordinates
      env = env_transform(rrij, at.Z[i], at.Z[j], Rs, Zs, calc.cutoff)
      # evaluate 
      Eij = evaluate(ace, env)
      E += Eij.val
   end
   return E
end


# function forces(calc::ACEBondPotential, at::Atoms)
#    F = zeros(SVector{3, Float64}, length(at))
#    for (i, j, rrij, Js, Rs, Zs) in bonds(at, calc)
#       Zi, Zj = at.Z[i], at.Z[j]
#       # find the right ace model 
#       ace = _get_model(calc, Zi, Zj)
#       # transform the euclidean to cylindrical coordinates
#       env = env_transform(rrij, Zi, Zj, Rs, Zs, calc.cutoff)
#       # evaluate 
#       dV_cyl = grad_config(ace, env)
#       # transform back? 
#       dV_drrij, dV_dRs = rrule_env_transform(rrij::SVector, Zi, Zj, Rs, Zs, dV_cyl, calc.cutoff)
#       # assemble the forces 
#       F[i] += dV_drrij 
#       F[j] -= dV_drrij 
#       for (k, dv) in zip(Js, dV_dRs)
#          F[k] -= dv
#          F[i] += 0.5 * dv 
#          F[j] += 0.5 * dv 
#       end
#    end
#    return F 
# end

# site_virial(dV::AbstractVector{JVec{T1}}, R::AbstractVector{JVec{T2}}
#             ) where {T1, T2} =  (
#       length(R) > 0 ? (- sum( dVi * Ri' for (dVi, Ri) in zip(dV, R) ))
#                     : zero(JMat{fltype_intersect(T1, T2)})
#       )

# function virial(calc::ACEBondPotential, at::Atoms{T}) where {T} 
#    V = zero(SMatrix{3, 3, T})
#    for (i, j, rrij, Js, Rs, Zs) in bonds(at, calc)
#       Zi, Zj = at.Z[i], at.Z[j]
#       # find the right ace model for this bond 
#       ace = _get_model(calc, Zi, Zj)
#       # transform the euclidean to cylindrical coordinates
#       env = env_transform(rrij, Zi, Zj, Rs, Zs, calc.cutoff)
#       # evaluate 
#       dV_cyl = grad_config(ace, env)
#       # transform back? 
#       dV_drrij, dV_dRs = rrule_env_transform(rrij::SVector, Zi, Zj, Rs, Zs, dV_cyl, calc.cutoff)
#       # assemble the virial  
#       #   dV_dRs contain derivative relative to midpoint 
#       #   dV_drrij contain derivative w.r.t. rrij
#       V -= dV_drrij * rrij'
#       for q = 1:length(dV_dRs)
#          V -= dV_dRs[q] * Rs[q]'
#       end
#    end
#    return V
# end



function energy(basis::ACEBondPotentialBasis, at::Atoms)
   E = zeros(Float64, length(basis))
   Et = zeros(Float64, length(basis))
   for (i, j, rrij, Js, Rs, Zs) in bonds(at, basis)
      # find the right ace model 
      ace = _get_model(basis, at.Z[i], at.Z[j])
      # transform the euclidean to cylindrical coordinates
      env = env_transform(rrij, at.Z[i], at.Z[j], Rs, Zs, basis.cutoff)
      # evaluate 
      ACE.evaluate!(Et, ace, ACE.ACEConfig(env))
      E += Et 
   end
   return E 
end


# function forces(basis::ACEBondPotentialBasis, at::Atoms)
#    F = zeros(SVector{3, Float64}, length(basis), length(at))
#    for (i, j, rrij, Js, Rs, Zs) in bonds(at, basis)
#       Zi, Zj = at.Z[i], at.Z[j]
#       # find the right ace model 
#       ace = _get_model(basis, Zi, Zj)
#       # transform the euclidean to cylindrical coordinates
#       env = env_transform(rrij, Zi, Zj, Rs, Zs, basis.cutoff)
#       # evaluate 
#       dB_cyl = evaluate_d(ace, env)
#       # transform back? 
#       dB_drrij, dB_dRs = rrule_env_transform(rrij, Zi, Zj, Rs, Zs, dB_cyl, basis.cutoff)
#       # assemble the forces 
#       F[:, i] += dB_drrij 
#       F[:, j] -= dB_drrij 
#       for n = 1:length(Js) 
#          k = Js[n]
#          dv = dB_dRs[:, n]
#          F[:, k] -= dv
#          F[:, i] += 0.5 * dv 
#          F[:, j] += 0.5 * dv 
#       end
#    end
#    return F  # TODO: this is probably the wrong format 
# end

# function virial(basis::ACEBondPotentialBasis, at::Atoms{T}) where {T} 
#    V = zeros(SMatrix{3, 3, T}, length(basis))
#    for (i, j, rrij, Js, Rs, Zs) in bonds(at, basis)
#       Zi, Zj = at.Z[i], at.Z[j]
#       # find the right ace model 
#       ace = _get_model(basis, Zi, Zj)
#       # transform the euclidean to cylindrical coordinates
#       env = env_transform(rrij, Zi, Zj, Rs, Zs, basis.cutoff)
#       # evaluate 
#       dB_cyl = evaluate_d(ace, env)
#       # transform back? 
#       dB_drrij, dB_dRs = rrule_env_transform(rrij, Zi, Zj, Rs, Zs, dB_cyl, basis.cutoff)
#       # assemble the virials
#       for iB = 1:length(basis)
#          V[iB] -= dB_drrij[iB] * rrij'
#          for q = 1:length(Rs)
#             V[iB] -= dB_dRs[iB, q] * Rs[q]'
#          end
#       end 
#    end
#    return V  # TODO: double-check the format 
# end
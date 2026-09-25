# The extension API of MatrixModels: a matrix-model type defined OUTSIDE the package
# (here: in this test module), using only public names, must work with the friction
# model, serialization and fitting layers. Guards the API against upstream changes.
using ACEfriction, Test, LinearAlgebra, SparseArrays, StaticArrays, Random
using ACEfriction: EuclideanMatrix, SphericalCutoff
import ACEfriction.MatrixModels as MM
import ACEfriction.MatrixModels: MatrixModel, matrix, basis, randf
import ACEbase.FIO: write_dict, read_dict
using ACEfriction.FrictionFit: weighted_l2_loss
using Flux: gradient, setup, Adam, update!
import AtomsBuilder: bulk, rattle!, set_elements
using AtomsBase: AbstractSystem

# ---- a model type built from the public API only ----------------------------------
# Σ = atom-centred pair blocks only (no onsite blocks); Γ = Σ Σᵀ (default structure).
struct OffsiteOnlyTestModel{O3S, SC} <: MatrixModel{O3S}
   offsite::MM.OffSiteModels
   n_rep::Int
   inds::MM.SiteInds
   id::Symbol
end
function OffsiteOnlyTestModel(offsite, id, sc::MM.SpeciesCoupling)
   O3S = typeof(first(values(offsite))).parameters[1]
   return OffsiteOnlyTestModel{O3S, typeof(sc)}(offsite, MM.n_rep(offsite), MM.site_inds(offsite), id)
end
_sc(::OffsiteOnlyTestModel{O3S, SC}) where {O3S, SC} = SC()

matrix(M::OffsiteOnlyTestModel, at::AbstractSystem; filter = (_, _) -> true, T = Float64) =
   MM.offsite_matrix(M.offsite, at; speciescoupling = _sc(M), filter = filter, T = T)
function basis(M::OffsiteOnlyTestModel, at::AbstractSystem; join_sites = false, filter = (_, _) -> true, T = Float64)
   B = MM.offsite_basis(M.offsite, at; inds = M.inds, speciescoupling = _sc(M), filter = filter, T = T)
   return join_sites ? B : (offsite = B,)
end
randf(::OffsiteOnlyTestModel, Σ::SparseMatrixCSC{SMatrix{3,3,T,9}}) where {T} = Σ * randn(SVector{3,T}, size(Σ, 2))
write_dict(M::OffsiteOnlyTestModel) =
   Dict("__id__" => "OffsiteOnlyTestModel", "offsite" => write_dict(M.offsite),
        "sc" => string(nameof(typeof(_sc(M)))), "id" => string(M.id))
read_dict(::Val{:OffsiteOnlyTestModel}, D::AbstractDict) =
   OffsiteOnlyTestModel(read_dict(D["offsite"]), Symbol(D["id"]), getfield(MM, Symbol(D["sc"]))())

# the same Σ, declared with the pairwise structure (Γ_ij = Σ_ij Σ_jiᵀ)
struct PairStructureTestModel{O3S} <: MatrixModel{O3S}
   inner::OffsiteOnlyTestModel{O3S}
   offsite::MM.OffSiteModels
   n_rep::Int
   inds::MM.SiteInds
   id::Symbol
end
PairStructureTestModel(m::OffsiteOnlyTestModel{O3S}) where {O3S} =
   PairStructureTestModel{O3S}(m, m.offsite, m.n_rep, m.inds, m.id)
MM.sigma_structure(::Type{<:PairStructureTestModel}) = MM.PairSigma()
matrix(M::PairStructureTestModel, at::AbstractSystem; kw...) = matrix(M.inner, at; kw...)
basis(M::PairStructureTestModel, at::AbstractSystem; kw...) = basis(M.inner, at; kw...)

# ---------------------------------------------------------------------------------
_dense(G, N) = (A = zeros(3N, 3N); for i = 1:N, j = 1:N; A[3i-2:3i, 3j-2:3j] .= G[i, j]; end; A)

@testset "extension API: matrix-model type defined outside the package" begin
   Random.seed!(3)
   at = set_elements(rattle!(bulk(:Cu, cubic = true) * 2, 0.1), [ rand((:Cu, :H)) for _ in 1:32 ])
   N = length(at)
   species = [:Cu, :H]
   bb = MM.offsite_linbasis(EuclideanMatrix(Float64), species; maxorder = 2, maxdeg = 4)
   for sc in (MM.SpeciesUnCoupled(), MM.SpeciesCoupled())
      offsite = MM.offsite_models(bb, SphericalCutoff(4.5), species, 2; speciescoupling = sc)
      @test Set(keys(offsite)) == (sc isa MM.SpeciesCoupled ? Set([(1,1), (1,29), (29,29)]) :
                                   Set([(1,1), (1,29), (29,1), (29,29)]))
      M = OffsiteOnlyTestModel(offsite, :ext, sc)
      fm = FrictionModel((ext = M,))

      # Σ equals the pair blocks of a CWC model with the same offsite coefficients
      cwc = CWCMatrixModel(EuclideanMatrix(Float64), species, species; maxorder = 2, maxdeg = 4, rcut = 4.5,
                           n_rep = 2, speciescoupling = sc)
      MM.set_params!(cwc, :offsite, MM.params(M; format = :matrix))
      Σ = matrix(M, at); Σc = matrix(cwc, at)
      @test maximum(norm(Σ[r][i, j] - Σc[r][i, j]) for r in 1:2, i in 1:N, j in 1:N if i != j) < 1e-12
      @test all(iszero(Σ[r][i, i]) for r in 1:2, i in 1:N)

      # basis · c == Σ, Γ = Σ Σᵀ, parameter plumbing
      B = basis(M, at; join_sites = true); c = MM.params(M; format = :native)
      @test maximum(norm(Matrix(sum(c[k][r] * B[k] for k in eachindex(B)) - Σ[r])) for r in 1:2) < 1e-10
      @test norm(_dense(Gamma(fm, at), N) - sum(_dense(Σ[r] * transpose(Σ[r]), N) for r in 1:2)) < 1e-10
      @test MM.nparams(M) == length(B)

      # serialization round trip
      fm2 = read_dict(write_dict(fm))
      @test norm(_dense(Gamma(fm2, at), N) - _dense(Gamma(fm, at), N)) < 1e-10

      # a declared structure is honoured (Γ as for the pairwise-coupled model)
      Mp = PairStructureTestModel(M)
      pwc = PWCMatrixModel(EuclideanMatrix(Float64), species, species; maxorder = 2, maxdeg = 4, rcut = 4.5,
                           n_rep = 2, speciescoupling = sc)
      MM.set_params!(pwc, MM.params(M; format = :matrix))
      @test norm(_dense(Gamma(FrictionModel((p = Mp,)), at), N) - _dense(Gamma(FrictionModel((p = pwc,)), at), N)) < 1e-10
   end

   # fitting: the Flux path picks the tensor layout / Γ from `sigma_structure`
   Random.seed!(5)
   offsite = MM.offsite_models(bb, SphericalCutoff(4.5), [:Cu, :H], 1; speciescoupling = MM.SpeciesCoupled())
   M = OffsiteOnlyTestModel(offsite, :ext, MM.SpeciesCoupled())
   for (name, model) in ("full" => M, "pair" => PairStructureTestModel(M))
      fm = FrictionModel((ext = model,))
      c_true = params(fm; format = :matrix, joinsites = true); set_params!(fm, c_true)
      ats = [ set_elements(rattle!(bulk(:Cu, cubic = true), 0.2), [:Cu, :H, :Cu, :H]) for _ in 1:3 ]
      fdata = [ FrictionData(a, Gamma(fm, a), collect(1:length(a))) for a in ats ]
      # the Flux model reproduces Γ of the reference coefficients exactly
      ffm = FluxFrictionModel(c_true)
      data = flux_assemble(fdata, fm, ffm)
      @test weighted_l2_loss(ffm, data) < 1e-16 * max(1, sum(sum(abs2, d.friction_tensor) for d in data))
      # and training from perturbed coefficients decreases the loss
      c0 = map(x -> x .+ 0.05 .* randn(size(x)), c_true)
      ffm0 = FluxFrictionModel(c0); loss0 = weighted_l2_loss(ffm0, data)
      opt = setup(Adam(0.01), ffm0)
      for _ in 1:30; update!(opt, ffm0, gradient(weighted_l2_loss, ffm0, data[1:2])[1]); end
      @testset "$name" begin
         @test weighted_l2_loss(ffm0, data) < loss0
      end
   end
end

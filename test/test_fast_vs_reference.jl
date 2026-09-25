# Consistency of the fast assembly (coefficient-contracted, per-centre evaluators;
# src/etbackend/fasteval.jl) with the previous implementation on MULTI-SPECIES systems.
#
# The reference re-implements the previous assembly loops literally: every block is
# obtained through the generic ET path (`evaluate_basis` -> Cartesian basis blocks ->
# contraction with the coefficients), one full evaluation per bond, no sharing and no
# folding. Both environment conventions (`partner_in_env` false/true) are covered.
using ACEfriction
using ACEfriction: EuclideanMatrix, SymmetricEuclideanMatrix, EuclideanVector, Invariant,
      SnowManCutoff, EllipsoidCutoff
import ACEfriction.MatrixModels
using ACEfriction.MatrixModels: _sites, _species, _mreduce, _keep_partner, evaluate_basis, _contract,
      _snowman_combine, _site_nb_table, _reverse_loc, _cwc_rcut, block_type, _get_SC, env_cutoff,
      SpeciesCoupled, SpeciesUnCoupled, Odd, NoZ2Sym
using ACEfriction.ETBackend: et_bonds
using Test, LinearAlgebra, StaticArrays, SparseArrays, Random
import AtomsBuilder: bulk, rattle!, set_elements
using AtomsBase: AbstractSystem

Random.seed!(2024)

# ---------------------------------------------------------------------------
# multi-species test systems
function _multispecies_system(sc::Int, species::Vector{Symbol}, r::Real)
   at0 = rattle!(bulk(:Cu, cubic = true) * sc, r)
   return set_elements(at0, [ species[rand(1:length(species))] for _ in 1:length(at0) ])
end
at3 = _multispecies_system(2, [:Cu, :Ni, :H], 0.15)          # 32 atoms, 3 species
at_small = _multispecies_system(1, [:Cu, :H], 0.1)             # 4 atoms: periodic self-images
_dense(G, N) = (A = zeros(3N, 3N); for i = 1:N, j = 1:N; A[3i-2:3i, 3j-2:3j] .= G[i, j]; end; A)
_denseΣ(Σ::AbstractMatrix{<:SMatrix}) = _dense(Σ, size(Σ, 1))
function _denseΣ(Σ::AbstractMatrix{<:SVector})
   N = size(Σ, 1); A = zeros(3N, N)
   for i = 1:N, j = 1:N; A[3i-2:3i, j] .= Σ[i, j]; end
   return A
end
_denseΣ(Σ::Diagonal) = _denseΣ(Matrix(Σ))

# ---------------------------------------------------------------------------
# reference assemblies (the previous algorithm: generic ET path, one evaluation per bond)
_zeroΣ(M, N) = [ zeros(block_type(first(values(getfield(M, hasfield(typeof(M), :onsite) ? :onsite : :offsite))).basis), N, N)
                 for _ in 1:M.n_rep ]

function ref_sigma(M::CWCMatrixModel, at::AbstractSystem; filter = (_, _) -> true)
   N = length(at); Z = _species(at); SC = _get_SC(M); Σ = _zeroΣ(M, N)
   for (i, neigs, Rs) in _sites(at, _cwc_rcut(M))
      (filter(i, at) && !isempty(neigs)) || continue
      Zs = Z[neigs]
      if haskey(M.onsite, Z[i])
         s = _contract(M.onsite[Z[i]], evaluate_basis(M.onsite[Z[i]], Rs, Zs))
         for r in 1:M.n_rep; Σ[r][i, i] += s[r]; end
      end
      for (jl, j) in enumerate(neigs)
         (filter(j, at) && _keep_partner(M.self_images, i, j)) || continue
         zz = _mreduce(Z[i], Z[j], SC); haskey(M.offsite, zz) || continue
         s = _contract(M.offsite[zz], evaluate_basis(M.offsite[zz], jl, Rs, Zs))
         for r in 1:M.n_rep; Σ[r][i, j] += s[r]; end
      end
   end
   return Σ
end

function ref_sigma(M::PWCMatrixModel{O3S, <:SphericalCutoff}, at::AbstractSystem; filter = (_, _) -> true) where {O3S}
   N = length(at); Z = _species(at); SC = _get_SC(M); Σ = _zeroΣ(M, N)
   for (i, neigs, Rs) in _sites(at, env_cutoff(M.offsite))
      (filter(i, at) && !isempty(neigs)) || continue
      Zs = Z[neigs]
      for (jl, j) in enumerate(neigs)
         (filter(j, at) && _keep_partner(M.self_images, i, j)) || continue
         zz = _mreduce(Z[i], Z[j], SC); haskey(M.offsite, zz) || continue
         s = _contract(M.offsite[zz], evaluate_basis(M.offsite[zz], jl, Rs, Zs))
         for r in 1:M.n_rep; Σ[r][i, j] += s[r]; end
      end
   end
   return Σ
end

# snowman: both bond ends evaluated on demand (the original two-evaluation walk)
function ref_sigma(M::PWCMatrixModel{O3S, <:SnowManCutoff}, at::AbstractSystem; filter = (_, _) -> true) where {O3S}
   N = length(at); Z = _species(at); SC = _get_SC(M); Σ = _zeroΣ(M, N)
   nb = _site_nb_table(at, env_cutoff(M.offsite))
   for i in 1:N
      (haskey(nb, i) && filter(i, at)) || continue
      (neigs_i, Rs_i) = nb[i]
      for (jl, j) in enumerate(neigs_i)
         (filter(j, at) && _keep_partner(M.self_images, i, j)) || continue
         zz = _mreduce(Z[i], Z[j], SC); haskey(M.offsite, zz) || continue
         om = M.offsite[zz]
         Bij = evaluate_basis(om, jl, Rs_i, Z[neigs_i])
         (neigs_j, Rs_j) = nb[j]
         il = _reverse_loc(neigs_j, Rs_j, i, Rs_i[jl])
         Bji = evaluate_basis(om, il, Rs_j, Z[neigs_j])
         s = _contract(om, _snowman_combine.(Ref(om.cutoff), Bij, Bji))
         for r in 1:M.n_rep; Σ[r][i, j] += s[r]; end
      end
   end
   return Σ
end

function ref_sigma(M::PWCMatrixModel{O3S, <:EllipsoidCutoff}, at::AbstractSystem; filter = (_, _) -> true) where {O3S}
   N = length(at); Z = _species(at); SC = _get_SC(M); Σ = _zeroΣ(M, N)
   om1 = first(values(M.offsite))
   for (i, j, rrij, _, Rs, Zs) in et_bonds(at, om1.cutoff)
      (filter(i, at) && filter(j, at) && _keep_partner(M.self_images, i, j)) || continue
      zz = _mreduce(Z[i], Z[j], SC); haskey(M.offsite, zz) || continue
      s = _contract(M.offsite[zz], evaluate_basis(M.offsite[zz], rrij, Rs, Zs))
      for r in 1:M.n_rep; Σ[r][i, j] += s[r]; end
   end
   return Σ
end

function ref_sigma(M::OnsiteOnlyMatrixModel, at::AbstractSystem; filter = (_, _) -> true)
   N = length(at); Z = _species(at); Σ = _zeroΣ(M, N)
   for (i, neigs, Rs) in _sites(at, env_cutoff(M.onsite))
      (haskey(M.onsite, Z[i]) && filter(i, at) && !isempty(neigs)) || continue
      s = _contract(M.onsite[Z[i]], evaluate_basis(M.onsite[Z[i]], Rs, Z[neigs]))
      for r in 1:M.n_rep; Σ[r][i, i] += s[r]; end
   end
   return Σ
end

# Γ from a reference Σ with the model's own coupling rule
function ref_gamma(M, Σref, N)
   Σs = [ M isa OnsiteOnlyMatrixModel ? Diagonal(diag(Σ)) : sparse(Σ) for Σ in Σref ]
   return _dense(Gamma(M, Σs), N)
end

# ---------------------------------------------------------------------------
"""check fast `matrix`/`basis`/`Gamma` against the reference for model `M` on system `at`."""
function check_consistent(M, at; filter = (_, _) -> true, tol = 1e-10)
   N = length(at)
   Σf = matrix(M, at; filter = filter); Σr = ref_sigma(M, at; filter = filter)
   scale = max(1.0, maximum(maximum(norm.(Σ)) for Σ in Σr))
   errΣ = maximum(norm(_denseΣ(Σf[r]) - _denseΣ(Σr[r])) for r in 1:M.n_rep) / scale
   @test errΣ < tol
   # un-contracted basis (fitting path) contracted with c reproduces Σ
   B = MatrixModels.basis(M, at; join_sites = true, filter = filter)
   c = params(M; format = :native)
   errB = maximum(norm(_denseΣ(sum(c[k][r] * B[k] for k in eachindex(B))) - _denseΣ(Σr[r])) for r in 1:M.n_rep) / scale
   @test errB < tol
   # friction tensor
   Gf = _dense(Gamma(FrictionModel((m = M,)), at; filter = filter), N)
   Gr = ref_gamma(M, Σr, N)
   @test norm(Gf - Gr) < tol * max(1.0, norm(Gr))
   return errΣ
end

const PROPS = (EuclideanMatrix(Float64), SymmetricEuclideanMatrix(Float64),
               EuclideanVector(Float64), Invariant(Float64))

@testset "fast vs reference on multi-species systems" begin
   species_env = [:Cu, :Ni, :H]; species_fr = [:Cu, :H]     # friction on a SUBSET of the species
   sel = (species_maxorder_dict = Dict(:H => 1),
          species_weight_cat = Dict(:H => 0.75, :Cu => 1.0, :Ni => 1.0))

   @testset "$(ACEfriction.ETBackend._property_str(prop)) partner_in_env=$pie" for prop in PROPS, pie in (false, true)
      # CWC: onsite + atom-centred offsite, species-uncoupled pairs
      m = CWCMatrixModel(prop, species_fr, species_env; maxorder = 2, maxdeg = 4, rcut = 4.5, n_rep = 2,
                         partner_in_env = pie, sel...)
      check_consistent(m, at3)
      # PWC spherical, species-coupled pairs
      m = PWCMatrixModel(prop, species_fr, species_env; maxorder = 2, maxdeg = 4, rcut = 4.5, n_rep = 2,
                         speciescoupling = SpeciesCoupled(), partner_in_env = pie, sel...)
      check_consistent(m, at3)
      # SnowMan, both symmetries
      for sym in (:symmetric, :antisymmetric)
         m = PWCMatrixModel(prop, species_fr, species_env, SnowManCutoff(4.5, sym; partner_in_env = pie);
                            maxorder = 2, maxdeg = 4, n_rep = 2, sel...)
         check_consistent(m, at3)
      end
   end

   @testset "onsite-only, maxorder 3, n_rep 3" begin
      m = OnsiteOnlyMatrixModel(EuclideanMatrix(Float64), species_fr, species_env; maxorder = 3, maxdeg = 5,
                                rcut = 4.5, n_rep = 3, sel...)
      check_consistent(m, at3)
   end

   @testset "ellipsoid PWC" begin
      m = PWCMatrixModel(EuclideanMatrix(Float64), species_fr, species_env, EllipsoidCutoff(3.0, 4.0, 5.0);
                         maxorder = 2, maxdeg = 4, n_rep = 2, sel...)
      check_consistent(m, at3)
   end

   @testset "Z2-odd vector pair model (momentum-conserving form)" begin
      m = PWCMatrixModel(EuclideanVector(Float64), species_fr, species_env; maxorder = 2, maxdeg = 4, rcut = 4.5,
                         n_rep = 2, z2sym = Odd(), partner_in_env = true, sel...)
      check_consistent(m, at3)
   end

   @testset "atom filter" begin
      m = CWCMatrixModel(EuclideanMatrix(Float64), species_fr, species_env; maxorder = 2, maxdeg = 4, rcut = 4.5, n_rep = 2)
      check_consistent(m, at3; filter = (i, at) -> isodd(i))
   end

   @testset "periodic self-images (include_self_images=$si) partner_in_env=$pie" for si in (false, true), pie in (false, true)
      m = CWCMatrixModel(EuclideanMatrix(Float64), [:Cu, :H], [:Cu, :H]; maxorder = 2, maxdeg = 4, rcut = 5.0,
                         n_rep = 2, include_self_images = si, partner_in_env = pie)
      check_consistent(m, at_small)
      m = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu, :H], [:Cu, :H]; maxorder = 2, maxdeg = 4, rcut = 5.0,
                         n_rep = 2, include_self_images = si, partner_in_env = pie)
      check_consistent(m, at_small)
      m = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu, :H], [:Cu, :H], SnowManCutoff(5.0, :symmetric; partner_in_env = pie);
                         maxorder = 2, maxdeg = 4, n_rep = 2, include_self_images = si)
      check_consistent(m, at_small)
   end

   @testset "coefficients changed after construction are picked up" begin
      m = CWCMatrixModel(EuclideanMatrix(Float64), species_fr, species_env; maxorder = 2, maxdeg = 4, rcut = 4.5, n_rep = 2)
      Σ1 = matrix(m, at3)
      θ = params(m; format = :matrix)
      set_params!(m, 0.3 .* θ)                      # through the public API
      check_consistent(m, at3)
      @test norm(_denseΣ(matrix(m, at3)[1]) - 0.3 * _denseΣ(Σ1[1])) < 1e-10 * norm(_denseΣ(Σ1[1]))
      # ... and when the coefficient vector is mutated in place behind the model's back
      c = params(first(values(m.offsite)))
      c .*= 2.0
      check_consistent(m, at3)
   end
end

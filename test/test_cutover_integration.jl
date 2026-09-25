# Integration test for the EquivariantTensors backend cutover: the public API
# (model constructors -> FrictionModel -> Gamma/Sigma, params round-trip, and the
# full Flux fitting path) on a real system, with no ACEfrictionCore.
using ACEfriction
using ACEfriction: EuclideanMatrix, SymmetricEuclideanMatrix, EllipsoidCutoff, SnowManCutoff
using ACEbase.FIO: write_dict, read_dict
using Test, LinearAlgebra, StaticArrays, SparseArrays
import AtomsBuilder: bulk, rattle!
using AtomsBase: ChemicalSpecies
using Flux: gradient, setup, Adam, update!
import Random

_dense(G, N) = (A = zeros(3N, 3N); for i=1:N, j=1:N; A[3i-2:3i, 3j-2:3j] .= G[i,j]; end; A)

@testset "cutover integration (ET backend, public API)" begin
   Random.seed!(1)
   at = rattle!(bulk(:Cu) * (2,2,2), 0.2); N = length(at)

   @testset "Gamma/Sigma PSD + params round-trip" begin
      m_on = OnsiteOnlyMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=2)
      m_pw = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=2)
      fm = FrictionModel((cov=m_on, equ=m_pw))
      Γ = Gamma(fm, at); Gd = _dense(Γ, N)
      @test norm(Gd - Gd') < 1e-8
      @test minimum(eigvals(Symmetric(Gd))) > -1e-8
      c = params(fm; format=:matrix, joinsites=true)
      @test keys(c) == (:cov, :equ)
      set_params!(fm, c)
      @test params(fm; format=:matrix, joinsites=true)[:cov] ≈ c[:cov]
   end

   @testset "ellipsoid PWC builds + PSD" begin
      m = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu], EllipsoidCutoff(3.0,4.0,5.0);
                         maxorder=2, maxdeg=4, n_rep=2, z2sym=ACEfriction.MatrixModels.NoZ2Sym())
      fm = FrictionModel((equ=m,))
      Γ = Gamma(fm, at); Gd = _dense(Γ, N)
      @test minimum(eigvals(Symmetric(Gd))) > -1e-8
   end

   @testset "SnowMan PWC ($sym): combine, basis/matrix consistency, IO" for sym in (:symmetric, :antisymmetric)
      # Σ_ij = c·basis(sphere_i, bond i→j) ± c·basis(sphere_j, bond j→i): combining both
      # bond ends makes Σ symmetric (+) or antisymmetric (−), selected by the cutoff's
      # type-parameter symmetry via _snowman_combine. Γ stays PSD; the un-contracted
      # basis contracted with c reproduces Σ; the symmetry round-trips through IO.
      m = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu], SnowManCutoff(5.0, sym);
                         maxorder=2, maxdeg=4, n_rep=2)
      fm = FrictionModel((equ=m,))
      Σ = Sigma(fm, at).equ[1]
      I, J, _ = findnz(Σ)
      resid = sym === :symmetric ? maximum(norm(Σ[i,j] - Σ[j,i]) for (i,j) in zip(I,J)) :
                                   maximum(norm(Σ[i,j] + Σ[j,i]) for (i,j) in zip(I,J))
      @test resid < 1e-10                                       # (anti)symmetry of Σ
      Gd = _dense(Gamma(fm, at), N)
      @test norm(Gd - Gd') < 1e-8
      @test minimum(eigvals(Symmetric(Gd))) > -1e-8
      # basis · c == matrix (fitting-path consistency)
      Boff = ACEfriction.MatrixModels.basis(m, at).offsite
      cc = m.offsite[(29,29)].c
      @test norm(sum(cc[k][1] * Boff[k] for k in eachindex(Boff)) - Σ) < 1e-10
      # cached (default) vs naive double-eval assembly agree exactly
      Σn = ACEfriction.MatrixModels.matrix(m, at; cache=false)
      Bn = ACEfriction.MatrixModels.basis(m, at; cache=false).offsite
      @test maximum(norm(Sigma(fm, at).equ[r] - Σn[r]) for r in eachindex(Σn)) < 1e-12
      @test maximum(norm(Boff[k] - Bn[k]) for k in eachindex(Boff)) < 1e-12
      # serialization round-trip (incl. the symmetry type parameter)
      fm2 = read_dict(write_dict(fm))
      @test fm2.matrixmodels.equ.offsite[(29,29)].cutoff isa SnowManCutoff{Float64, sym}
      @test norm(_dense(Gamma(fm2, at), N) - Gd) < 1e-10
   end

   @testset "partner_in_env (factorised pair environment): consistency + IO" begin
      # With the bond partner pooled into the environment the pair blocks factorise and
      # are assembled per centre; the result must still equal basis·c, be PSD, and the
      # option must survive serialization. It is a different basis from the default.
      for M in (CWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=2, partner_in_env=true),
                PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=2, partner_in_env=true),
                PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu], SnowManCutoff(5.0, :symmetric; partner_in_env=true);
                               maxorder=2, maxdeg=4, n_rep=2))
         om = first(values(M.offsite))
         @test ACEfriction.ETBackend.partner_in_env(om.cutoff)
         fm = FrictionModel((equ=M,))
         Σ = Sigma(fm, at).equ
         B = ACEfriction.MatrixModels.basis(M, at; join_sites=true)
         c = params(M; format=:native)
         @test maximum(norm(sum(c[k][r] * B[k] for k in eachindex(B)) - Σ[r]) for r in 1:2) < 1e-10
         Gd = _dense(Gamma(fm, at), N)
         @test minimum(eigvals(Symmetric(Gd))) > -1e-8
         fm2 = read_dict(write_dict(fm))
         @test ACEfriction.ETBackend.partner_in_env(first(values(fm2.matrixmodels.equ.offsite)).cutoff)
         @test norm(_dense(Gamma(fm2, at), N) - Gd) < 1e-10
      end
      # partner excluded (original convention) and factorised are different bases
      M0 = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=1, partner_in_env=false)
      M1 = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=1, partner_in_env=true)
      set_params!(M1, params(M0))
      @test norm(Sigma(FrictionModel((e=M0,)), at).e[1] - Sigma(FrictionModel((e=M1,)), at).e[1]) > 1e-6
   end

   @testset "partner_in_env: default for new models, preserved for saved ones" begin
      pie(M) = all(ACEfriction.ETBackend.partner_in_env(om.cutoff) for om in values(M.offsite))
      # newly built atom-centred models pool the partner into the environment by default
      @test ACEfriction.SphericalCutoff(5.0).partner_in_env && SnowManCutoff(5.0, :antisymmetric).partner_in_env
      @test pie(PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0))
      @test pie(PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu], SnowManCutoff(5.0, :antisymmetric); maxorder=2, maxdeg=4))
      @test pie(CWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0))
      @test pie(CWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu], ACEfriction.MatrixModels.AtomCentered(); maxorder_on=2, maxdeg_on=4))
      # a model saved before the option existed (no "partner_in_env" key) keeps the
      # original partner-excluded basis on loading
      M0 = CWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=2, partner_in_env=false)
      fm0 = FrictionModel((equ=M0,))
      strip_pie!(d) = d
      strip_pie!(d::AbstractDict) = (delete!(d, "partner_in_env"); foreach(strip_pie!, values(d)); d)
      D = strip_pie!(write_dict(fm0))
      fm_old = read_dict(D)
      @test !pie(fm_old.matrixmodels.equ)
      @test norm(_dense(Gamma(fm_old, at), N) - _dense(Gamma(fm0, at), N)) < 1e-10
      # an explicitly saved setting (either value) round-trips
      for flag in (false, true)
         Mf = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, partner_in_env=flag)
         @test pie(read_dict(write_dict(FrictionModel((e=Mf,)))).matrixmodels.e) == flag
      end
   end

   @testset "Flux fitting path: loss decreases" begin
      m_on = OnsiteOnlyMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=2)
      m_pw = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=2)
      fm = FrictionModel((cov=m_on, equ=m_pw))
      c_true = params(fm; format=:matrix, joinsites=true); set_params!(fm, c_true)
      ats = [ rattle!(bulk(:Cu) * (2,2,2), 0.3) for _ in 1:4 ]
      fdata = [ FrictionData(a, Gamma(fm, a), collect(1:length(a))) for a in ats ]
      c0 = map(x -> 0.01 .* randn(size(x)), c_true); set_params!(fm, c0)
      ffm = FluxFrictionModel(c0)
      data = flux_assemble(fdata, fm, ffm)
      loss0 = weighted_l2_loss(ffm, data)
      opt = setup(Adam(0.01), ffm)
      for _ in 1:50
         ∂L = gradient(weighted_l2_loss, ffm, data)[1]; update!(opt, ffm, ∂L)
      end
      loss1 = weighted_l2_loss(ffm, data)
      @test isfinite(loss1) && loss1 < loss0
   end
end

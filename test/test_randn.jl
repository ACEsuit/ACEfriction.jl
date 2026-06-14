# `randf(fm, Σ)` draws a Normal(0, Γ) pseudo-random force from the diffusion matrices
# Σ = Sigma(fm, at), where Γ = Gamma(fm, at) is the friction tensor. This test checks
# that property empirically: over many samples the sample covariance of the stacked 3N
# force vector converges to the dense Γ.
#
# Covered for every concrete matrix model (CWC, PWC, OnsiteOnly) in both O3-symmetry
# flavours (matrix-equivariant SMatrix{3,3} / vector-equivariant SVector{3} blocks) and
# under both self-image policies. The cell is small enough that PWC sees periodic
# self-images, so `include_self_images` actually changes Σ here:
#   - ExcludeSelfImages (default): PWC Σ is purely off-diagonal (Σ_ii == 0).
#   - IncludeSelfImages: PWC acquires diagonal Σ_ii blocks (contributing 1·Σ_ii Σ_iiᵀ).
# In both cases Cov(randf) must equal Γ.
using ACEfriction
using ACEfriction: EuclideanMatrix, EuclideanVector, ExcludeSelfImages, IncludeSelfImages
using ACEbase.FIO: write_dict, read_dict
using Test, LinearAlgebra, StaticArrays
import AtomsBuilder: bulk, rattle!
import Random

# stack a length-N Vector{SVector{3}} into a length-3N Vector
_flat(R) = reduce(vcat, R)
# densify the sparse/Diagonal block matrix Γ into a 3N×3N Matrix
_dense(G, N) = (A = zeros(3N, 3N); for i=1:N, j=1:N; A[3i-2:3i, 3j-2:3j] .= G[i,j]; end; A)

@testset "randf: empirical covariance ≈ Γ" begin
    Random.seed!(1)
    # small cell that gives the PWC model self-image (diagonal) Σ blocks when included
    at = rattle!(bulk(:Cu) * (2,1,1), 0.2); N = length(at)
    n = 400_000

    models = [
        ("OnsiteOnly", (p, inc) -> OnsiteOnlyMatrixModel(p, [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=2, include_self_images=inc)),
        ("PWC",        (p, inc) -> PWCMatrixModel(p, [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=2, include_self_images=inc)),
        ("CWC",        (p, inc) -> CWCMatrixModel(p, [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=2, include_self_images=inc)),
    ]
    properties = [("matrix-equ", EuclideanMatrix(Float64)), ("vector-equ", EuclideanVector(Float64))]

    @testset "$mname ($pname, include_self_images=$inc)" for (mname, build) in models,
                                                              (pname, prop) in properties,
                                                              inc in (false, true)
        fm = FrictionModel((m = build(prop, inc),))
        # randomize parameters so Γ is a nontrivial (non-zero) target
        c = params(fm; format=:matrix, joinsites=true)
        set_params!(fm, map(x -> randn(size(x)), c))

        Σ = Sigma(fm, at)
        Γd = _dense(Gamma(fm, Σ), N)

        # PWC: the policy controls whether self-image diagonal Σ blocks exist
        if mname == "PWC"
            Σ1 = Σ.m[1]
            ndiag = count(i -> norm(Σ1[i, i]) > 1e-12, 1:N)
            inc ? (@test ndiag > 0) : (@test ndiag == 0)
        end

        # accumulate the empirical (zero-mean) covariance of the stacked force
        C = zeros(3N, 3N)
        for _ in 1:n
            f = _flat(randf(fm, Σ))
            C .+= f * f'
        end
        C ./= n

        # relative Frobenius error; sampling error of a covariance estimate ~ 1/√n
        @test norm(C - Γd) / norm(Γd) < 0.05
    end

    @testset "self-image policy round-trips through IO" begin
        for inc in (false, true)
            m = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, include_self_images=inc)
            fm = FrictionModel((m = m,))
            fm2 = read_dict(write_dict(fm))
            P = inc ? IncludeSelfImages : ExcludeSelfImages
            @test fm2.matrixmodels.m.self_images isa P
        end
    end
end

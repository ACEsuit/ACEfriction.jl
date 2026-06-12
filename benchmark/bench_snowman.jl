# Benchmark: cached vs naive (double-eval) directed-bond assembly for the SnowMan PWC
# matrix model. Standalone and dependency-free — run with
#
#     julia --project=. benchmark/bench_snowman.jl
#
# The SnowMan block is Σ_ij = c·B(sphere_i, bond i→j) ± c·B(sphere_j, bond j→i). The
# naive path (`cache=false`) evaluates each directed-bond ACE basis twice (once per
# ordered pair that uses it); the cached path (`cache=true`, the default) memoises the
# per-directed-bond evaluation and reuses it, halving the ACE work. This script first
# asserts the two paths agree numerically, then times `matrix` and `basis` for both.

using ACEfriction, LinearAlgebra, SparseArrays, StaticArrays
using ACEfriction.FrictionModels
using ACEfriction: EuclideanMatrix, SnowManCutoff, EllipsoidCutoff
import ACEfriction.MatrixModels: matrix, basis
import AtomsBuilder: bulk, rattle!
import Random

# minimum wall-clock over `samples` runs, after one warm-up (returns seconds)
function _best(f; samples = 5)
    f()                                            # warm-up / compile
    minimum(@elapsed(f()) for _ in 1:samples)
end

# count assembled directed bonds (== number of per-centre ACE evaluations in the
# cached path; the naive path does twice this many)
_ndirected(m, at) = nnz(first(matrix(m, at)))      # one replica's sparsity = #pairs/bonds

const SIZES = [(1,1,1), (2,2,2), (3,3,3), (4,4,4)]

function run_bench(; sym = :symmetric, maxorder = 2, maxdeg = 6, n_rep = 2, samples = 5)
    println("\nSnowMan PWC benchmark  (symmetry = :$sym, maxorder=$maxorder, ",
            "maxdeg=$maxdeg, n_rep=$n_rep)")
    println(rpad("N", 6), rpad("#pairs", 9),
            rpad("matrix naive", 14), rpad("matrix cached", 15), rpad("speedup", 9),
            rpad("basis naive", 14), rpad("basis cached", 15), "speedup")
    println("-"^88)
    Random.seed!(1)
    m = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu], SnowManCutoff(5.0, sym);
                       maxorder = maxorder, maxdeg = maxdeg, n_rep = n_rep)
    for sz in SIZES
        at = rattle!(bulk(:Cu) * sz, 0.2); N = length(at)

        # ---- agreement check (cached must equal naive) ----
        Σc = matrix(m, at; cache = true);  Σn = matrix(m, at; cache = false)
        Bc = basis(m, at; cache = true).offsite;  Bn = basis(m, at; cache = false).offsite
        dΣ = maximum(norm(Σc[r] - Σn[r]) for r in eachindex(Σc))
        dB = maximum(norm(Bc[k] - Bn[k]) for k in eachindex(Bc))
        @assert dΣ < 1e-12 && dB < 1e-12 "cached/naive disagree (N=$N): dΣ=$dΣ dB=$dB"

        npairs = _ndirected(m, at)
        tmn = _best(() -> matrix(m, at; cache = false); samples = samples)
        tmc = _best(() -> matrix(m, at; cache = true);  samples = samples)
        tbn = _best(() -> basis(m, at;  cache = false); samples = samples)
        tbc = _best(() -> basis(m, at;  cache = true);  samples = samples)
        ms(x) = round(1e3 * x; digits = 2)
        sp(a, b) = round(a / b; digits = 2)
        println(rpad(N, 6), rpad(npairs, 9),
                rpad(ms(tmn), 14), rpad(ms(tmc), 15), rpad(sp(tmn, tmc), 9),
                rpad(ms(tbn), 14), rpad(ms(tbc), 15), sp(tbn, tbc))
    end
end

# Reference: ellipsoid (bond-centred) PWC model. It assembles via the `et_bonds`
# iterator and evaluates each bond exactly once — there is no double-eval and hence no
# `cache` toggle. Timed here as a point of comparison against the snowman numbers.
function run_bench_ellipsoid(; maxorder = 2, maxdeg = 6, n_rep = 2, samples = 5)
    println("\nEllipsoid PWC benchmark  (single implementation, no cache toggle; ",
            "maxorder=$maxorder, maxdeg=$maxdeg, n_rep=$n_rep)")
    println(rpad("N", 6), rpad("#bonds", 9), rpad("matrix (ms)", 14), "basis (ms)")
    println("-"^44)
    Random.seed!(1)
    m = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu], EllipsoidCutoff(5.0, 5.0, 5.0);
                       maxorder = maxorder, maxdeg = maxdeg, n_rep = n_rep)
    for sz in SIZES
        at = rattle!(bulk(:Cu) * sz, 0.2); N = length(at)
        nbonds = _ndirected(m, at)
        tm = _best(() -> matrix(m, at); samples = samples)
        tb = _best(() -> basis(m, at);  samples = samples)
        ms(x) = round(1e3 * x; digits = 2)
        println(rpad(N, 6), rpad(nbonds, 9), rpad(ms(tm), 14), ms(tb))
    end
end

run_bench(sym = :symmetric)
run_bench(sym = :antisymmetric)
run_bench_ellipsoid()
println("\n(times are min wall-clock in ms over samples; speedup = naive / cached)")

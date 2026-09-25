# Benchmark: cost of evaluating friction models (`matrix`, i.e. Σ, and the
# un-contracted `basis` used for fitting) for all coupling schemes, per atom and
# relative to an energy-like evaluation of an ACE basis of the same size.
# Standalone and dependency-free — run with
#
#     julia --project=. benchmark/bench_evaluation.jl
#
# The "energy-like" reference evaluates an invariant (scalar) site basis with the same
# body order / degree / cutoff through the same fast evaluator, on the same
# neighbourhoods — the friction analogue of an ACE site-energy evaluation (radial and
# angular embeddings, pooling, one pass over the product basis). All models use
# matrix-valued blocks. `PWC sph. (excl)` is the atom-centred pair model with the
# original, partner-excluded bond environment (`partner_in_env = false`), whose pair
# blocks cannot share per-centre work; every other atom-centred model uses the default
# `partner_in_env = true`.
#
# Times are single-threaded minimum wall-clock over a few samples. Edit the settings
# below to change system size, body order / degree, or the number of replicas.

using ACEfriction, LinearAlgebra, StaticArrays, Printf, Random
using ACEfriction: EuclideanMatrix, SnowManCutoff, EllipsoidCutoff
import ACEfriction.MatrixModels as MM
import ACEfriction.ETBackend as ETB
import AtomsBuilder: bulk, rattle!

# ---- settings ----
const CELL    = 2                     # Cu supercell: bulk(:Cu, cubic=true) * CELL  (CELL=2: 32 atoms)
const RCUT    = 5.0                   # atom-centred cutoff (Å)
const ELLIPS  = EllipsoidCutoff(3.5, 4.0, 5.0)
const N_REP   = 2
const SIZES   = [(2, 5), (3, 6)]      # (maxorder, maxdeg)
const BASIS   = true                  # also time `basis` (the fitting path)
const SAMPLES = 3

# minimum wall-clock over `samples` runs, after one warm-up (seconds); a full GC before
# each run keeps collections triggered by earlier (allocation-heavy) runs out of it
function _best(f; samples = SAMPLES)
    f()
    minimum(begin GC.gc(); @elapsed(f()) end for _ in 1:samples)
end

# energy-like reference: invariant scalar basis of the same size through the fast evaluator
function energy_like(at, maxorder, maxdeg)
    b = MM.onsite_linbasis(ETB.ETInvariant(), [:Cu]; rcut = RCUT, maxorder = maxorder, maxdeg = maxdeg)
    fm = ETB.ETFastModel(b, [ SVector(x) for x in randn(length(b)) ])
    Z = MM._species(at); sd = ETB.ETSiteData(fm)
    return () -> begin
        s = 0.0
        for (_, neigs, Rs) in MM._sites(at, RCUT)
            s += ETB.evaluate!(sd, fm, Rs, Z[neigs])[1][1]
        end
        s
    end
end

function models(maxorder, maxdeg)
    kw = (maxorder = maxorder, maxdeg = maxdeg, n_rep = N_REP)
    P = EuclideanMatrix(Float64)
    return [
        "Onsite"          => OnsiteOnlyMatrixModel(P, [:Cu], [:Cu]; rcut = RCUT, kw...),
        "CWC"             => CWCMatrixModel(P, [:Cu], [:Cu]; rcut = RCUT, kw...),
        "PWC sph."        => PWCMatrixModel(P, [:Cu], [:Cu]; rcut = RCUT, kw...),
        "PWC sph. (excl)" => PWCMatrixModel(P, [:Cu], [:Cu]; rcut = RCUT, partner_in_env = false, kw...),
        "PWC snowman"     => PWCMatrixModel(P, [:Cu], [:Cu], SnowManCutoff(RCUT, :antisymmetric); kw...),
        "PWC ellipsoid"   => PWCMatrixModel(P, [:Cu], [:Cu], ELLIPS; kw...),
    ]
end

function run_bench()
    Random.seed!(1)
    at = rattle!(bulk(:Cu, cubic = true) * CELL, 0.1); N = length(at)
    println("Friction-model evaluation cost: $N Cu atoms, rcut = $RCUT Å, n_rep = $N_REP, ",
            "$(Threads.nthreads()) thread(s); times in μs per atom")
    for (mo, md) in SIZES
        tE = _best(energy_like(at, mo, md))
        println("\nmaxorder = $mo, maxdeg = $md      energy-like: ", @sprintf("%.1f", 1e6 * tE / N), " μs/atom")
        @printf("%-17s %12s %10s %14s\n", "model", "matrix", "× energy", BASIS ? "basis" : "")
        println("-"^56)
        for (name, m) in models(mo, md)
            tm = _best(() -> MM.matrix(m, at))
            tb = BASIS ? _best(() -> MM.basis(m, at); samples = 2) : NaN
            @printf("%-17s %12.1f %10.1f %14s\n", name, 1e6 * tm / N, tm / tE,
                    BASIS ? @sprintf("%.1f", 1e6 * tb / N) : "")
        end
    end
end

run_bench()

# Thread safety of the matrix models: concurrent `matrix` / `basis` calls on ONE
# model object (e.g. threaded trajectory ensembles sharing a friction model) must
# give the same result as serial calls. Guards against evaluation buffers stored in
# the model (the fast evaluators keep all workspaces per call).
#
# The check needs several threads. When the test process runs single-threaded (the
# `Pkg.test` default), this file re-runs itself in a `julia -t 4` subprocess.
using Test

if Threads.nthreads() == 1 && get(ENV, "ACEFRICTION_THREADTEST_CHILD", "") != "1"
   @testset "thread safety (subprocess, 4 threads)" begin
      cmd = `$(Base.julia_cmd()) -t 4 --startup-file=no --project=$(Base.active_project()) $(@__FILE__)`
      env = copy(ENV); env["ACEFRICTION_THREADTEST_CHILD"] = "1"
      env["JULIA_LOAD_PATH"] = join(LOAD_PATH, Sys.iswindows() ? ";" : ":")
      @test success(pipeline(setenv(cmd, env); stdout = stdout, stderr = stderr))
   end
else

using ACEfriction, LinearAlgebra, StaticArrays, SparseArrays, Random
using ACEfriction: EuclideanMatrix, SnowManCutoff, EllipsoidCutoff
import ACEfriction.MatrixModels as MM
import AtomsBuilder: bulk, rattle!, set_elements

@info "thread-safety test running on $(Threads.nthreads()) threads"
Random.seed!(7)
_sys() = set_elements(rattle!(bulk(:Cu, cubic = true) * 2, 0.1),
                      [ rand((:Cu, :H)) for _ in 1:32 ])
ats = [ _sys() for _ in 1:6 ]
kw = (maxorder = 2, maxdeg = 4, n_rep = 2)
models = [
   "onsite"          => OnsiteOnlyMatrixModel(EuclideanMatrix(Float64), [:Cu, :H], [:Cu, :H]; rcut = 4.5, kw...),
   "CWC"             => CWCMatrixModel(EuclideanMatrix(Float64), [:Cu, :H], [:Cu, :H]; rcut = 4.5, kw...),
   "PWC spherical"   => PWCMatrixModel(EuclideanMatrix(Float64), [:Cu, :H], [:Cu, :H]; rcut = 4.5, kw...),
   "PWC sph. (pie)"  => PWCMatrixModel(EuclideanMatrix(Float64), [:Cu, :H], [:Cu, :H]; rcut = 4.5,
                                       partner_in_env = true, kw...),
   "PWC snowman"     => PWCMatrixModel(EuclideanMatrix(Float64), [:Cu, :H], [:Cu, :H],
                                       SnowManCutoff(4.5, :antisymmetric); kw...),
   "PWC ellipsoid"   => PWCMatrixModel(EuclideanMatrix(Float64), [:Cu, :H], [:Cu, :H],
                                       EllipsoidCutoff(3.5, 4.0, 4.5); kw...),
]

_maxdiff(A, B) = maximum(maximum(norm.(A[r] - B[r]); init = 0.0) for r in eachindex(B))

@testset "thread safety: concurrent matrix/basis on one model" begin
   for (name, m) in models
      Σref = [ MM.matrix(m, at) for at in ats ]
      Bref = [ MM.basis(m, at; join_sites = true) for at in ats ]
      tasks = [ Threads.@spawn begin
                   a = mod1(k, length(ats))
                   (a, MM.matrix(m, ats[a]), isodd(k) ? MM.basis(m, ats[a]; join_sites = true) : nothing)
                end for k in 1:48 ]
      errΣ = 0.0; errB = 0.0
      for t in tasks
         a, Σ, B = fetch(t)
         errΣ = max(errΣ, _maxdiff(Σ, Σref[a]))
         B === nothing || (errB = max(errB, _maxdiff(B, Bref[a])))
      end
      @testset "$name" begin
         @test errΣ < 1e-12
         @test errB < 1e-12
      end
   end
end

end # threaded body

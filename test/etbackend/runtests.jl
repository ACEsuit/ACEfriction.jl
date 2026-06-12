# Aggregate runner for the ET backend test suite.
# Run:  julia --project=. test/etbackend/runtests.jl
#
# Each test file is included in its own fresh module to avoid top-level name
# clashes (several files define helpers like `randenv`, `species`, `transform`).
using Test

const _ETB_TESTS = [
   "test_output_transforms.jl",
   "test_radial.jl",
   "test_sitebasis.jl",
   "test_models.jl",
   "test_friction_models.jl",
   "test_bondbasis.jl",
   "test_selection.jl",
   "test_offsite.jl",
   "test_basis_fit.jl",
   "test_serialization.jl",
   # test_equivalence.jl is intentionally omitted post-cutover: it cross-checked the
   # old ACEfrictionCore backend, which no longer exists.
]

@testset "ETBackend suite" begin
   for f in _ETB_TESTS
      @info "running $f"
      m = Module(Symbol("ETB_", replace(f, ".jl" => "")))
      # bare modules have no `include`; inject one so nested includes work
      Base.eval(m, :(include(p) = Base.include(@__MODULE__, p)))
      Base.include(m, joinpath(@__DIR__, f))
   end
end

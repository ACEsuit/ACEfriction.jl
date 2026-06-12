using Test

# Each test file is run in its own fresh module so top-level globals and (re-)included
# ETBackend type identities don't leak between files.
function _run_test(f)
    m = Module(Symbol("Test_", replace(f, r"[./]" => "_")))
    Base.eval(m, :(include(p) = Base.include(@__MODULE__, p)))
    Base.include(m, joinpath(@__DIR__, f))
end

@testset "ACEfriction.jl" begin
    # EquivariantTensors backend unit suite (bases, models, serialization, fitting format)
    @testset "ET backend" begin _run_test("etbackend/runtests.jl") end

    # public-API integration on the ET backend (constructors -> Gamma/Sigma -> Flux fit)
    @testset "cutover integration" begin _run_test("test_cutover_integration.jl") end

    # I/O data round-trip
    @testset "I/O data" begin _run_test("test_IO_data.jl") end

    # model-fit tests on real data (build -> save/load -> Flux fit to tolerance)
    @testset "CWC model fit" begin _run_test("test_ac_model_fit.jl") end
    @testset "PWC fit (spherical cutoff)" begin _run_test("test_pwcsc_model_fit.jl") end
    @testset "PWC fit (ellipsoid cutoff)" begin _run_test("test_pwcec_model_fit.jl") end
end

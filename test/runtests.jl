

using ACEfriction, Test, LinearAlgebra, StaticArrays

using ACEfriction.DataUtils: FrictionData
using ACEfriction.FrictionFit
using Flux
using Flux.MLUtils
using ACEfriction.MatrixModels
using ACEfriction: RWCMatrixModel
using ACEfrictionCore

##
@testset "ACEfriction.jl" begin
    # ------------------------------------------
    #@testset "Basic test" begin include("test_model_evaluation.jl") end
    # @testset "I/O models" begin include("./test_IO_models.jl") end
    @testset "I/O data" begin include("./test_IO_data.jl") end
    @testset "test ac model fit" begin include("./test_ac_model_fit.jl") end
    @testset "test pwc model fit spherical cutoff" begin include("./test_pwcsc_model_fit.jl") end
    @testset "test pwc model fit elliptical cutoff" begin include("./test_pwcec_model_fit.jl") end
     

end


    
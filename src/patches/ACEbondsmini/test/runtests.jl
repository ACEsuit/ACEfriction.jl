using ACEbonds
using Test

@testset "ACEbonds.jl" begin
    # Write your tests here.
    # @testset "Bonds basics" begin include("test_bonds.jl"); end 
    # @testset "Invariant Cylindrical" begin include("test_invcyl.jl"); end
    @testset "Calculator (Cylindrical env.)" begin include("test_cylindricalbondpot.jl"); end
    @testset "Calculator (Ellipsoid env.)" begin include("test_ellipsoidbondpot.jl"); end
    # @testset "Bond Iterators" begin include("test_bonditerators.jl"); end
end

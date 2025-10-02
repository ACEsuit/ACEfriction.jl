using ACE
using ACE.Testing
using ACE.ACEbonds.BondCutoffs
using ACEfriction
using ACEfriction.AtomCutoffs
using ACEfriction.FrictionModels
using ACEfriction.MatrixModels
using ACEfriction: RWCMatrixModel
using Distributions: Categorical
using JuLIP
using LinearAlgebra
using Test





#%%
@info "Testing write_dict and read_dict for PWCMatrixModel with SphericalCutoff"
fm_pwcsc2 = ACE.read_dict(ACE.write_dict(fm_pwcsc));
for _ in 1:5
    at = gen_config([:H,:Cu], n_min=2,n_max=2, species_prop = Dict(:H=>.5, :Cu=>.5), species_min = Dict(:H=>1, :Cu=>1),  maxnit = 1000)
    print_tf(@test Gamma(fm_pwcsc,at) == Gamma(fm_pwcsc2,at))
end
println()

@info "Testing save_dict and load_dict for PWCMatrixModel with SphericalCutoff"
tmpname = string(tempname(),".json")
save_dict(tmpname, write_dict(fm_pwcsc))
fm_pwcsc2 = read_dict(load_dict(tmpname))
for _ in 1:5
    at = gen_config([:H,:Cu], n_min=2,n_max=2, species_prop = Dict(:H=>.5, :Cu=>.5), species_min = Dict(:H=>1, :Cu=>1),  maxnit = 1000)
    print_tf(@test norm(Gamma(fm_pwcsc,at) - Gamma(fm_pwcsc2,at))< tol)
end
println()

#%%
@info "Testing write_dict and read_dict for PWCMatrixModel with EllipsoidCutoff"
fm_pwcec2 = ACE.read_dict(ACE.write_dict(fm_pwcec));
for _ in 1:5
    at = gen_config([:H,:Cu], n_min=2,n_max=2, species_prop = Dict(:H=>.5, :Cu=>.5), species_min = Dict(:H=>1, :Cu=>1),  maxnit = 1000)
    print_tf(@test Gamma(fm_pwcec,at) == Gamma(fm_pwcec2,at))
end
println()

@info "Testing save_dict and load_dict for PWCMatrixModel with EllipsoidCutoff"
tmpname = string(tempname(),".json")
save_dict(tmpname, write_dict(fm_pwcec))
fm_pwcec2 = read_dict(load_dict(tmpname))
for _ in 1:5
    at = gen_config([:H,:Cu], n_min=2,n_max=2, species_prop = Dict(:H=>.5, :Cu=>.5), species_min = Dict(:H=>1, :Cu=>1),  maxnit = 1000)
    print_tf(@test norm(Gamma(fm_pwcec,at) - Gamma(fm_pwcec2,at))< tol)
end
println()

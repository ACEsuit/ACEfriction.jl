using ACEfriction
using ACEfriction.MatrixModels
using Test
using ACE.Testing
using JuLIP
using Distributions: Categorical
using LinearAlgebra: norm
using Random
using Flux
using Flux.MLUtils

function gen_config(species; n_min=2,n_max=2, species_prop = Dict(z=>1.0/length(species) for z in species), species_min = Dict(z=>1 for z in keys(species_prop)),  maxnit = 1000)
    species = collect(keys(species_prop))
    n = rand(n_min:n_max)
    at = rattle!(bulk(:Cu, cubic=true) * n, 0.3)
    N_atoms = length(at)
    d = Categorical( values(species_prop)|> collect)
    nit = 0
    while true 
        at.Z = AtomicNumber.(species[rand(d,N_atoms)]) 
        if all(sum(at.Z .== AtomicNumber(z)) >= n_min  for (z,n_min) in species_min)
            break
        elseif nit > maxnit 
            @error "Number of iterations exceeded $maxnit."
            exit()
        end
        nit+=1
    end
    return at
end


train_tol = .03;
tol = 1E-9;

@info "Create PWC+OO friction model with SphericalCutoff"
species_friction = [:H]
species_env = [:Cu,:H]
species_substrat = [:Cu]
rcut = 5.0

m_equ_on =  OnsiteOnlyMatrixModel(EuclideanMatrix(Float64), species_friction, species_env; species_substrat=species_substrat, id=:equ0, n_rep = 3, rcut = rcut, maxorder=2, maxdeg=3,
species_maxorder_dict = Dict( :H => 1), 
species_weight_cat = Dict(:H => .75, :Cu=> 1.0)
);

m_equ_off = PWCMatrixModel(EuclideanMatrix(Float64),species_friction,species_env;
        z2sym= NoZ2Sym(),
        speciescoupling = SpeciesUnCoupled(),
        species_substrat = species_substrat,
        n_rep = 3,
        maxorder=2, 
        maxdeg=5, 
        rcut = rcut, 
        r0_ratio=.4, 
        rin_ratio=.04, 
        species_maxorder_dict = Dict( :H => 0), 
        species_weight_cat = Dict(:H => 1.0, :Cu=> 1.0),
        bond_weight = 1.0
    );
fm= FrictionModel((mequ_off=m_equ_off, m_equ_on=m_equ_on)); 


@info "Testing save_dict and load_dict"
tmpname = string(tempname(),".json")
save_dict(tmpname, write_dict(fm))
fm2 = read_dict(load_dict(tmpname))
for _ in 1:5
    at = gen_config([:H,:Cu], n_min=2,n_max=2, species_prop = Dict(:H=>.5, :Cu=>.5), species_min = Dict(:H=>1, :Cu=>1),  maxnit = 1000)
    print_tf(@test norm(Gamma(fm,at) - Gamma(fm2,at))< tol)
end
println()

@info "Load test data"
fname = "/test/test-data-100"
filename = string(pkgdir(ACEfriction),fname,".h5")
rdata = ACEfriction.DataUtils.load_h5fdata(filename); 

# Partition data into train and test set and convert to 
rng = MersenneTwister(12)
shuffle!(rng, rdata)
n_train = Int(ceil(.8 * length(rdata)))
n_test = length(rdata) - n_train

fdata = Dict("train" => rdata[1:n_train], 
            "test"=> rdata[n_train+1:end]);
            
@info "Fit friction model"  

c = params(fm;format=:matrix, joinsites=true)

ffm = FluxFrictionModel(c)
set_params!(ffm; sigma=1E-8)

# Create preprocessed data including basis evaluations that can be used to fit the model
flux_data = Dict( "train"=> flux_assemble(fdata["train"], fm, ffm),
                  "test"=> flux_assemble(fdata["test"], fm, ffm));



loss_traj = Dict("train"=>Float64[], "test" => Float64[])

epoch = 0
batchsize = 10
nepochs = 300

opt = Flux.setup(Adam(1E-3, (0.99, 0.999)),ffm)
dloader = DataLoader(flux_data["train"], batchsize=batchsize, shuffle=true)

@info "Starting training"
for _ in 1:nepochs
    global epoch
    epoch+=1
    for d in dloader
        ∂L∂m = Flux.gradient(weighted_l2_loss,ffm, d)[1]
        Flux.update!(opt,ffm, ∂L∂m)       # method for "explicit" gradient
    end
    for tt in ["test","train"]
        push!(loss_traj[tt], weighted_l2_loss(ffm,flux_data[tt]))
    end
    # println("Epoch: $epoch, Abs avg Training Loss: $(loss_traj["train"][end]/n_train)), Test Loss: $(loss_traj["test"][end]/n_test))")
end
# println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end]), Test Loss: $(loss_traj["test"][end])")
println("Epoch: $epoch, Avg Training Loss: $(loss_traj["train"][end]/n_train), Test Loss: $(loss_traj["test"][end]/n_test)")

@test minimum(loss_traj["train"]/n_train) < train_tol

set_params!(fm, params(ffm))


for d in fdata["train"]
    Σ = Sigma(fm, d.atoms)
@test norm(Gamma(fm, Σ) - Gamma(fm, d.atoms)) < tol
end





using LinearAlgebra
using ACEfriction.FrictionModels
using ACEfrictionCore: scaling, params
using ACEfriction
using ACEfriction.FrictionFit
using ACEfriction.DataUtils
using Flux
using Flux.MLUtils
using ACEfrictionCore
using Random
using ACEfriction.FrictionFit

using ACEfriction.MatrixModels
using CUDA


cuda = CUDA.functional()



rdata_train = ACEfriction.DataUtils.load_h5fdata("./examples/data/dpd-train-x.h5"); 
rdata_test = ACEfriction.DataUtils.load_h5fdata("./examples/data/dpd-train-x.h5"); 

fdata = Dict("train" => FrictionData.(rdata_train), 
            "test"=> FrictionData.(rdata_test));
(n_train, n_test) = length(fdata["train"]), length(fdata["test"])

using ACEfriction.AtomCutoffs: SphericalCutoff
using ACEfriction.MatrixModels: NoZ2Sym, SpeciesUnCoupled



m_cov = mbdpd_matrixmodel(EuclideanVector(), [:X], [:X];
    maxorder=1, 
    maxdeg=8,    
    rcutbond = 5.0, 
    rcutenv = 5.0,
    zcutenv = 5.0,
    n_rep = 1, 
    species_substrat=[], 
    # Not documented:   
    r0_ratio=0.4, 
    rin_ratio=.00, 
    )

fm= FrictionModel((m_cov=m_cov,)); 
                                            
c = params(fm;format=:matrix, joinsites=true)
ffm = FluxFrictionModel(c)



# Create preprocessed data including basis evaluations that can be used to fit the model
flux_data = Dict( "train"=> flux_assemble(fdata["train"], fm, ffm),
                  "test"=> flux_assemble(fdata["test"], fm, ffm));



#if CUDA is available, convert relevant arrays to cuarrays
using CUDA
cuda = CUDA.functional()

if cuda
    ffm = fmap(cu, ffm)
end

loss_traj = Dict("train"=>Float64[], "test" => Float64[])

epoch = 0
batchsize = 10
nepochs = 100

opt = Flux.setup(Adam(1E-2, (0.999, 0.999)),ffm)
dloader = cuda ? DataLoader(flux_data["train"] |> gpu, batchsize=batchsize, shuffle=true) : DataLoader(flux_data["train"], batchsize=batchsize, shuffle=true)

using ACEfriction.FrictionFit: weighted_l2_loss, weighted_l1_loss

for _ in 1:nepochs
    epoch+=1
    @time for d in dloader
        ∂L∂m = Flux.gradient(weighted_l2_loss,ffm, d)[1]
        Flux.update!(opt,ffm, ∂L∂m)       # method for "explicit" gradient
    end
    for tt in ["test","train"]
        push!(loss_traj[tt], weighted_l2_loss(ffm,flux_data[tt]))
    end
    println("Epoch: $epoch, Abs avg Training Loss: $(loss_traj["train"][end]/n_train)), Test Loss: $(loss_traj["test"][end]/n_test))")
end
println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end]), Test Loss: $(loss_traj["test"][end])")
println("Epoch: $epoch, Avg Training Loss: $(loss_traj["train"][end]/n_train), Test Loss: $(loss_traj["test"][end]/n_test)")

minimum(loss_traj["train"]/n_train) <0.01
set_params!(fm, params(ffm))

#%%
at = fdata["test"][1].atoms
@time Gamma(fm, at);
@time Σ = Sigma(fm, at);
@time Gamma(fm, Σ)
@time randf(fm, Σ)



using AtomsBuilder: bulk, rattle!, set_elements
using AtomsBase: atomic_number
using Distributions: Categorical

function gen_config(species; n_min=2,n_max=2, species_prop = Dict(z=>1.0/length(species) for z in species), species_min = Dict(z=>1 for z in keys(species_prop)),  maxnit = 1000)
    species = collect(keys(species_prop))
    n = rand(n_min:n_max)
    at0 = rattle!(bulk(:Cu, cubic=true) * n, 0.3)
    N_atoms = length(at0)
    d = Categorical( values(species_prop)|> collect)
    nit = 0
    while true
        # species are element Symbols; rebuild the system with the drawn species
        # (AtomsBase systems are immutable, unlike JuLIP's `at.Z = ...`).
        at = set_elements(at0, species[rand(d,N_atoms)])
        if all(sum(atomic_number(at, :) .== atomic_number(z)) >= n_min  for (z,n_min) in species_min)
            return at
        elseif nit > maxnit
            @error "Number of iterations exceeded $maxnit."
            exit()
        end
        nit+=1
    end
end

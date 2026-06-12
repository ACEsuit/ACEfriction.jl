# Serialization for ET backend types (write_dict / read_dict).
#
# Bases are serialized by their *construction recipe* (stored in `basis.meta`) and
# rebuilt with `onsite_basis` / `bond_basis` on read — we do not dump the ET tensor
# or P4ML objects. Models add their coefficient vector. The dicts use the same
# "__id__" convention as ACEfrictionCore so they slot into the friction-model
# serialization at cut-over.

using StaticArrays
# extend the canonical ACEbase serialization functions (the ones ACEfriction /
# DataUtils / save_dict-load_dict use), so all `write_dict`/`read_dict` methods
# across the package live on one generic function.
import ACEbase.FIO: read_dict, write_dict

# ---- site basis (onsite or bond) : rebuild from recipe ----

# reconstruct the selection kwargs (weight, p_sel, per-species dicts) from a recipe
function _selection_kwargs(sel::AbstractDict)
   w = sel["weight"]
   return (weight = Dict(:n => w["n"], :l => w["l"]),
           p_sel = sel["p_sel"],
           species_weight_cat = Dict(sel["weight_cat"]),
           species_minorder_dict = Dict(sel["minorder_dict"]),
           species_maxorder_dict = Dict(sel["maxorder_dict"]))
end

"""rebuild an `ETFrictionSiteBasis` from a serialized construction recipe."""
function rebuild_basis(recipe::AbstractDict)
   prop = _property_from_str(recipe["property"])
   species = Int.(recipe["species"])
   radial = Dict(Symbol(k) => v for (k, v) in recipe["radial"])
   sel = _selection_kwargs(recipe["selection"])
   kind = recipe["kind"]
   if kind == "onsite"
      return onsite_basis(prop, species;
               rcut = recipe["rcut"], maxorder = recipe["maxorder"],
               maxdeg = recipe["maxdeg"], maxl = recipe["maxl"],
               sel..., radial...)
   elseif kind == "bond"
      return bond_basis(prop, species;
               z2sym = Symbol(recipe["z2sym"]),
               rcut = recipe["rcut"], maxorder = recipe["maxorder"],
               maxdeg = recipe["maxdeg"], maxl = recipe["maxl"],
               bond_weight = recipe["selection"]["bond_weight"],
               sel..., radial...)
   else
      error("unknown basis recipe kind = $kind")
   end
end

write_dict(b::ETFrictionSiteBasis) =
      Dict{String, Any}("__id__" => "ETBackend_SiteBasis",
                        "recipe" => b.meta["recipe"])

read_dict(::Val{:ETBackend_SiteBasis}, D::AbstractDict) = rebuild_basis(D["recipe"])

# ---- cutoffs ----
write_dict(c::SphericalCutoff) =
      Dict{String,Any}("__id__" => "ETBackend_SphericalCutoff", "rcut" => c.rcut)
read_dict(::Val{:ETBackend_SphericalCutoff}, D::AbstractDict) = SphericalCutoff(Float64(D["rcut"]))

write_dict(c::SnowManCutoff) =
      Dict{String,Any}("__id__" => "ETBackend_SnowManCutoff", "rcut" => c.rcut,
                       "symmetry" => String(symmetry(c)))
read_dict(::Val{:ETBackend_SnowManCutoff}, D::AbstractDict) =
      SnowManCutoff(Float64(D["rcut"]), Symbol(get(D, "symmetry", "symmetric")))

write_dict(c::EllipsoidCutoff) =
      Dict{String,Any}("__id__" => "ETBackend_EllipsoidCutoff",
            "rcutbond" => c.rcutbond, "rcutenv" => c.rcutenv, "zcutenv" => c.zcutenv)
read_dict(::Val{:ETBackend_EllipsoidCutoff}, D::AbstractDict) =
      EllipsoidCutoff(Float64(D["rcutbond"]), Float64(D["rcutenv"]), Float64(D["zcutenv"]))

# ---- flattened onsite model : recipe + coefficients ----

function write_dict(m::ETOnsiteModel{NR}) where {NR}
   return Dict{String, Any}(
      "__id__" => "ETBackend_OnsiteModel",
      "basis" => write_dict(m.basis),
      "n_rep" => NR,
      "c" => collect(reinterpret(Float64, m.c)))
end

function read_dict(::Val{:ETBackend_OnsiteModel}, D::AbstractDict)
   basis = read_dict(D["basis"])
   NR = Int(D["n_rep"])
   cflat = Vector{Float64}(D["c"])
   c = collect(reinterpret(SVector{NR, Float64}, cflat))
   return ETOnsiteModel(basis, c)
end

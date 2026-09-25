# Fit-quality comparison of the two bond-environment conventions of the atom-centred
# pair models: `partner_in_env = true` (default; the bond partner is pooled into the
# bond environment, which makes all pair blocks of a centre factorise) versus
# `partner_in_env = false` (the original, partner-excluded environment).
#
# Both conventions are fitted with identical seeds, initial coefficients, basis size
# and ordering, and training settings (Adam 1e-3, batch size 10, as in the package fit
# tests), on the H/Cu data `test/test-data-100.h5`. Reported: relative Γ errors
# ‖Γ_fit − Γ‖ / ‖Γ‖ on train and test data at the final and the best epoch, split into
# diagonal and off-diagonal blocks, and the median per-configuration relative error.
#
#     julia --project=. benchmark/fit_partner_in_env.jl [options]
#
# Options (defaults in brackets):
#     --models=cwc,pwc,snowman   models to compare                          [cwc,pwc,snowman]
#     --maxorder=2 --maxdeg=5    bond-basis body order / degree                          [2, 5]
#     --seeds=1,2,3              initialisation / shuffling seeds (with --cv: the folds) [1,2,3]
#     --epochs=1500              training epochs                                         [1500]
#     --cv=5                     k-fold cross-validation instead of the 80/20 split         [off]
#     --relative                 weight each configuration by 1/‖Γ‖² (scale-balanced loss) [off]
#     --out=FILE.csv             append one row per fit             [fit_partner_in_env.csv]
#     --summarize A.csv B.csv …  only print the paired summary of existing result files
#
# The fits are independent, so large comparisons can be split over several processes
# (e.g. one per model or per seed, each with its own --out) and merged with
# --summarize. A full default run takes roughly 15–20 minutes single-threaded.
#
# Notes on the H/Cu data: friction acts on the two H atoms only, so every bond is H–H.
# The models below allow H in the bond environment; with H excluded from the
# environment factors (as in the package fit tests) the two conventions coincide
# exactly. They also coincide exactly for the antisymmetric snowman at maxorder = 2.
# ‖Γ‖ spans five orders of magnitude over the configurations and five of them carry
# >99% of Σ‖Γ‖², so the absolute loss is dominated by a few configurations; --relative
# balances them.

using ACEfriction, LinearAlgebra, Random, Statistics, Printf
using ACEfriction: EuclideanMatrix, SnowManCutoff
using ACEfriction.MatrixModels: AtomCentered, NoZ2Sym, SpeciesUnCoupled
using ACEfriction.FrictionFit: weighted_l2_loss
using Flux, Flux.MLUtils

# ---------------------------------------------------------------------------
# options

function parse_options(args)
   opt = Dict{String, Any}("models" => ["cwc", "pwc", "snowman"], "maxorder" => 2, "maxdeg" => 5,
                           "seeds" => [1, 2, 3], "epochs" => 1500, "cv" => 0, "relative" => false,
                           "out" => "fit_partner_in_env.csv", "summarize" => String[])
   i = 1
   while i <= length(args)
      a = args[i]
      if a == "--summarize"
         opt["summarize"] = args[i+1:end]; break
      elseif a == "--relative"
         opt["relative"] = true
      elseif startswith(a, "--") && occursin('=', a)
         k, v = split(a[3:end], '=', limit = 2)
         k in ("models",)             ? (opt[k] = String.(split(v, ','))) :
         k in ("seeds",)              ? (opt[k] = parse.(Int, split(v, ','))) :
         k in ("maxorder", "maxdeg", "epochs", "cv") ? (opt[k] = parse(Int, v)) :
         k == "out"                   ? (opt[k] = String(v)) :
         error("unknown option --$k")
      else
         error("unknown argument $a (see the header of this script)")
      end
      i += 1
   end
   return opt
end

# ---------------------------------------------------------------------------
# data and models

function load_data()
   rdata = ACEfriction.DataUtils.load_h5fdata(joinpath(pkgdir(ACEfriction), "test", "test-data-100.h5"))
   shuffle!(MersenneTwister(12), rdata)                 # as in the package fit tests
   return rdata
end

# train / test split: 80/20 as in the package tests, or fold `k` of `nfold` as test set
function split_data(rdata, nfold, k)
   nfold == 0 && (ntr = Int(ceil(0.8 * length(rdata))); return rdata[1:ntr], rdata[ntr+1:end])
   nf = length(rdata) ÷ nfold
   tidx = ((k - 1) * nf + 1):(k * nf)
   return rdata[setdiff(eachindex(rdata), tidx)], rdata[tidx]
end

# H/Cu models of the package fit tests, but with H allowed in the bond environment
function build_model(kind::String, pie::Bool, maxorder::Int, maxdeg::Int)
   sf, se = [:H], [:Cu, :H]
   on() = OnsiteOnlyMatrixModel(EuclideanMatrix(Float64), sf, se; species_substrat = [:Cu], id = :on,
            n_rep = 3, rcut = 5.0, maxorder = maxorder, maxdeg = 3, species_maxorder_dict = Dict(:H => 1),
            species_weight_cat = Dict(:H => 0.75, :Cu => 1.0))
   offkw = (species_substrat = [:Cu], n_rep = 3, maxorder = maxorder, maxdeg = maxdeg, r0_ratio = 0.4,
            rin_ratio = 0.04, species_weight_cat = Dict(:H => 1.0, :Cu => 1.0), bond_weight = 1.0)
   if kind == "cwc"
      m = CWCMatrixModel(EuclideanMatrix(Float64), sf, se, AtomCentered(); species_substrat = [:Cu], id = :cwc,
            n_rep = 3, rcut_on = 5.0, rcut_off = 5.0, maxorder_on = maxorder, maxdeg_on = 3,
            maxorder_off = maxorder, maxdeg_off = maxdeg, species_maxorder_dict_on = Dict(:H => 1),
            species_weight_cat_on = Dict(:H => 0.75, :Cu => 1.0),
            species_weight_cat_off = Dict(:H => 1.0, :Cu => 1.0), bond_weight = 0.5, partner_in_env = pie)
      return FrictionModel((cwc = m,))
   elseif kind == "pwc"
      off = PWCMatrixModel(EuclideanMatrix(Float64), sf, se; id = :off, rcut = 5.0, z2sym = NoZ2Sym(),
            speciescoupling = SpeciesUnCoupled(), partner_in_env = pie, offkw...)
   elseif kind == "snowman"
      # antisymmetric: off-diagonal blocks Γ_ij = -Σ_ij Σ_ijᵀ ⪯ 0, as in the data
      off = PWCMatrixModel(EuclideanMatrix(Float64), sf, se, SnowManCutoff(5.0, :antisymmetric; partner_in_env = pie);
            id = :off, offkw...)
   else
      error("unknown model $kind (cwc, pwc, snowman)")
   end
   return FrictionModel((off = off, on = on()))
end

# ---------------------------------------------------------------------------
# metrics

_normΓ(x) = sqrt(sum(norm(x.friction_tensor[i, j])^2 for i in x.friction_indices, j in x.friction_indices))

# relative Γ errors over a data set: (all, diagonal blocks, off-diagonal blocks)
function relerr(ffm, data)
   e = zeros(3); n = zeros(3)
   for d in data
      G = ffm(d.B, d.Tfm); Gt = d.friction_tensor
      for i in axes(G, 3), j in axes(G, 4)
         de = sum(abs2, view(G, :, :, i, j) .- view(Gt, :, :, i, j)); dn = sum(abs2, view(Gt, :, :, i, j))
         k = i == j ? 2 : 3
         e[1] += de; n[1] += dn; e[k] += de; n[k] += dn
      end
   end
   return sqrt.(e ./ max.(n, eps()))
end

relerr_cfg(ffm, data) = [ sqrt(sum(abs2, ffm(d.B, d.Tfm) .- d.friction_tensor) / sum(abs2, d.friction_tensor))
                          for d in data ]

# ---------------------------------------------------------------------------
# fitting

const COLUMNS = ["model", "maxorder", "maxdeg", "weighting", "split", "partner_in_env", "seed", "nparams",
                 "epochs", "train", "test", "test_diag", "test_offdiag", "best_test", "best_epoch",
                 "train_cfg_median", "test_cfg_median", "best_test_cfg_median"]

function fit_once(fm, fdata, seed, nepochs)
   ffm = FluxFrictionModel(params(fm))
   Random.seed!(1000 + seed); set_params!(ffm; sigma = 1e-8)          # identical init for both conventions
   opt = Flux.setup(Adam(1e-3, (0.99, 0.999)), ffm)
   Random.seed!(2000 + seed)
   dl = DataLoader(fdata["train"], batchsize = 10, shuffle = true)
   best = (Inf, 0); bestc = Inf
   for ep in 1:nepochs
      for d in dl
         Flux.update!(opt, ffm, Flux.gradient(weighted_l2_loss, ffm, d)[1])
      end
      if ep % 25 == 0 || ep == nepochs
         te = relerr(ffm, fdata["test"])[1]
         te < best[1] && (best = (te, ep))
         bestc = min(bestc, median(relerr_cfg(ffm, fdata["test"])))
      end
   end
   tr = relerr(ffm, fdata["train"]); te = relerr(ffm, fdata["test"])
   return (train = tr[1], test = te[1], test_diag = te[2], test_offdiag = te[3], best_test = best[1],
           best_epoch = best[2], train_cfg_median = median(relerr_cfg(ffm, fdata["train"])),
           test_cfg_median = median(relerr_cfg(ffm, fdata["test"])), best_test_cfg_median = bestc)
end

function run_comparison(opt)
   rdata = load_data()
   out = opt["out"]
   isfile(out) || open(io -> println(io, join(COLUMNS, ",")), out, "w")
   weighting = opt["relative"] ? "relative" : "absolute"
   splitname = opt["cv"] == 0 ? "80/20" : "cv$(opt["cv"])"
   w(data) = Dict("observations" => (opt["relative"] ? [1 / _normΓ(x)^2 for x in data] : ones(length(data))),
                  "diag" => 2.0, "sub_diag" => 1.0, "off_diag" => 1.0)
   for kind in opt["models"], pie in (false, true)
      Random.seed!(1)
      fm = build_model(kind, pie, opt["maxorder"], opt["maxdeg"])
      ffm0 = FluxFrictionModel(params(fm))
      np = sum(length, values(params(fm)))
      assembled = Dict{Int, Any}()                 # fold -> assembled data (one fold for 80/20)
      for seed in opt["seeds"]
         fold = opt["cv"] == 0 ? 1 : seed
         fdata = get!(assembled, fold) do
            train, test = split_data(rdata, opt["cv"], fold)
            Dict("train" => flux_assemble(train, fm, ffm0; weights = w(train)),
                 "test"  => flux_assemble(test, fm, ffm0; weights = w(test)))
         end
         t = @elapsed r = fit_once(fm, fdata, seed, opt["epochs"])
         row = Any[kind, opt["maxorder"], opt["maxdeg"], weighting, splitname, pie, seed, np, opt["epochs"], r...]
         open(io -> println(io, join(row, ",")), out, "a")
         @printf("%-8s partner_in_env=%-5s seed=%d  train %.4f  test %.4f  best test %.4f @%d  [%.0f s]\n",
                 kind, pie, seed, r.train, r.test, r.best_test, r.best_epoch, t)
         flush(stdout)
      end
   end
   return out
end

# ---------------------------------------------------------------------------
# paired summary (means over seeds / folds; per-seed differences true − false)

function summarize(files)
   rows = Dict{String, String}[]
   for f in files, (l, line) in enumerate(eachline(f))
      l == 1 && continue
      line == join(COLUMNS, ",") && continue                          # repeated header
      push!(rows, Dict(zip(COLUMNS, split(line, ','))))
   end
   groups = Dict{NTuple{5, String}, Vector{Dict{String, String}}}()
   for r in rows
      push!(get!(groups, (r["model"], r["maxorder"], r["maxdeg"], r["weighting"], r["split"]), []), r)
   end
   metrics = ["train", "best_test", "test", "test_diag", "test_offdiag", "train_cfg_median",
              "best_test_cfg_median"]
   for (key, rs) in sort(collect(groups), by = first)
      println("\n== $(key[1])  maxorder=$(key[2]) maxdeg=$(key[3])  loss: $(key[4])  split: $(key[5])  ",
              "nparams=$(rs[1]["nparams"])")
      @printf("%-22s %18s %18s   %s\n", "relative error", "partner excluded", "partner in env", "paired diff (in − excl)")
      for m in metrics
         a = Dict(r["seed"] => parse(Float64, r[m]) for r in rs if r["partner_in_env"] == "false")
         b = Dict(r["seed"] => parse(Float64, r[m]) for r in rs if r["partner_in_env"] == "true")
         fmt(d) = isempty(d) ? "-" : @sprintf("%.4f ± %.4f", mean(values(d)), std(collect(values(d)); corrected = false))
         diffs = [ b[s] - a[s] for s in sort(collect(keys(a))) if haskey(b, s) ]
         @printf("%-22s %18s %18s   %s\n", m, fmt(a), fmt(b), join((@sprintf("%+.4f", d) for d in diffs), " "))
      end
   end
end

# ---------------------------------------------------------------------------

opt = parse_options(ARGS)
if isempty(opt["summarize"])
   summarize([run_comparison(opt)])
else
   summarize(opt["summarize"])
end

#
# Draft ACEModel Interface: 
#
# * A model incorporates both the parameterisation and the parameters. 
# * E.g. a LinearACEModel would know about the basis and the coefficients
# * We can then perform the following operations: 
#    - evaluate the model at a given configuration, with given parameters
#    - take gradient w.r.t. the configuration 
#    - take gradient w.r.t. the parameters 
#
# An advantage will be that we can do the parameter reduction trick and then 
# use a fast evaluator to obtain the gradient w.r.t. configuration which is 
# very expensive if we naively take derivatives w.r.t. the basis and then 
# apply parameters. (this is not implemented below and will be the next step)
# Another advantage is that model, and full model construction are all stored 
# together for future inspection.

"""
`struct LinearACEModel`: linear model for symmetric properties in terms of 
a `SymmetricBasis`. 

The typical way to construct a linear model is to first construct a basis 
`basis`, some default coefficients `c` and then call 
```julia
model = LinearACEModel(basis, c)
```

### Multiple properties 

If `c::Vector{<: Number}` then the output of the model will be the property 
encoded in the basis. But one can also use a single basis to produce 
multiple properties (with different coefficients). This can be achieved by 
simply supplying `c::Vector{SVector{N, T}}` where `N` will then be the 
number of properties. 
"""
struct LinearACEModel{TB, TP, TEV} <: AbstractACEModel 
   basis::TB
   c::Vector{TP}
   evaluator::TEV   
end 

struct NaiveEvaluator end 

function LinearACEModel(basis::SymmetricBasis, c = zeros(length(basis)); 
               evaluator = :standard) 
   if evaluator == :naive 
      ev = NaiveEvaluator()
   elseif evaluator == :standard 
      ev = ProductEvaluator(basis, c) 
   elseif evaluator == :recursive 
      error("Recursive evaluator not yet implemented")
   else 
      error("unknown evaluator")
   end
   return LinearACEModel(basis, c, ev)
end

# LinearACEModel(basis::SymmetricBasis, c::Vector, evaluator) = 
#          LinearACEModel(basis, c, ev, VectorPool{eltype(c)})

# ------- parameter wrangling 

nparams(m::LinearACEModel) = length(m.c)

params(m::LinearACEModel) = copy(m.c)

function set_params!(m::LinearACEModel, c) 
   m.c[:] .= c
   set_params!(m.evaluator, m.basis, c)
   return m 
end

set_params!(::NaiveEvaluator, args...) = nothing 

# ------------------- FIO

==(V1::LinearACEModel, V2::LinearACEModel) = 
      _allfieldsequal(V1, V2)

write_dict(V::LinearACEModel) = 
      Dict( "__id__" => "ACE_LinearACEModel", 
             "basis" => write_dict(V.basis), 
                 "c" => write_dict(V.c), 
         "evaluator" => write_dict(V.evaluator) )

function read_dict(::Val{:ACE_LinearACEModel}, D::Dict) 
   basis = read_dict(D["basis"])
   c = read_dict(D["c"])
   # special evaluator version of the read_dict 
   evaluator = read_dict(Val(Symbol(D["evaluator"]["__id__"])), 
                         D["evaluator"], basis, c)
   return LinearACEModel(basis, c, evaluator)
end

write_dict(ev::NaiveEvaluator) = 
      Dict("__id__" => "ACE_NaiveEvaluator" )

read_dict(::Val{:ACE_NaiveEvaluator}, D::Dict, args...) = 
      NaiveEvaluator()





# ------------------- dispatching on the evaluators 

evaluate(m::LinearACEModel, X::AbstractConfiguration) = 
      evaluate(m, m.evaluator, X)

# grad_config function removed - derivative functionality has been removed

# grad_params function removed - derivative functionality has been removed

# grad_params_config function removed - derivative functionality has been removed



      # TODO: fix terminology, bring in linear with the _rrule_.... thing 
adjoint_EVAL_D(m::LinearACEModel, cfg::AbstractConfiguration, w) = 
      adjoint_EVAL_D(m, m.evaluator, cfg, w)


# ------------------- implementation of naive evaluator 
#  this is only intended for testing, as it uses the naive evaluation of  
#  the symmetric basis, rather than the conversion to the AA basis

function evaluate(m::LinearACEModel, ::NaiveEvaluator, cfg::AbstractConfiguration)  
   B = evaluate(m.basis, cfg)
   val = contract(m.c, B)
   release!(B)
   return val 
end 

# grad_config function for NaiveEvaluator removed - derivative functionality has been removed

function adjoint_EVAL_D(m::LinearACEModel, ::NaiveEvaluator, 
                        X::AbstractConfiguration, w) 
   error("grad_params_config functionality has been removed")
   # dB = grad_params_config(m, X)  -- removed
   # g = zeros(size(dB, 1))  -- removed
   # for i = 1:length(g), j = 1:size(dB, 2)  -- removed
   #    g[i] += dot(dB[i, j], w[j])  -- removed
   # end  -- removed
   # release_dB!(m.basis, dB)  -- removed
   # return g  -- removed
end


# ------------------- an rrule for evaluating a linear model

# ChainRules imports removed - derivative functionality has been removed


function _adj_evaluate(dp, model::ACE.LinearACEModel, cfg)
   error("grad_params functionality has been removed")
   # gp_ = ACE.grad_params(model, cfg)  -- removed
   # gp = [ a * dp for a in gp_ ]  -- removed
   # return NoTangent(), gp, _rrule_evaluate(dp, model, cfg)  -- removed
end

# ChainRules.rrule function removed - derivative functionality has been removed


# ChainRules.rrule function for _adj_evaluate removed - derivative functionality has been removed

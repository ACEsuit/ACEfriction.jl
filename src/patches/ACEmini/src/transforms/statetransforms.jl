using StaticArrays
import ACE 
import ACE: evaluate, 
            write_dict, read_dict, 
            DState
            # frule_evaluate! 

using LinearAlgebra: I, norm  

# TODO: 
#   - retire GetNorm
#   - polish GetVal, and expand to GetVali 
#   - polish the frule and rrule implementations 

# ------------------ Some different ways to produce an argument 

abstract type StateTransform end 
abstract type StaticGet <: StateTransform end 

ACE.evaluate(fval::StaticGet, X) = getval(X, fval)

valtype(fval::StaticGet, X) = typeof(evaluate(fval, X))


struct GetVal{VSYM} <: StaticGet end 

getval(X, ::GetVal{VSYM}) where {VSYM} = getproperty(X, VSYM) 

_one(x::Number) = one(x)
_one(x::SVector{3, T}) where {T}  = SMatrix{3, 3, T}(I)

get_symbols(::GetVal{VSYM}) where {VSYM} = (VSYM,)


      
# TODO - this is incomplete for now 
# struct GetVali{VSYM, IND} <: StaticGet end 
# getval(X, ::GetVali{VSYM, IND}) where {VSYM, IND} = getproperty(X, VSYM)[IND]
# getval_d(X, ::GetVali{VSYM, IND}) where {VSYM, IND} = __e(getproperty(X, VSYM), Val{IND}())


struct GetNorm{VSYM} <: StaticGet end 

getval(X, ::GetNorm{VSYM}) where {VSYM} = norm(getproperty(X, VSYM))

function getval_d(X, ::GetNorm{VSYM}) where {VSYM}
   x = getproperty(X, VSYM)
   return DState( NamedTuple{(VSYM,)}( (x/norm(x),) ) )
end 


get_symbols(::GetNorm{VSYM}) where {VSYM} = (VSYM,)


write_dict(fval::StaticGet) = Dict("__id__" => "ACE_StaticGet", 
                                   "expr" => string(typeof(fval)) )

read_dict(::Val{:ACE_StaticGet}, D::Dict) = eval( Meta.parse(D["expr"]) )()


# rrule_evaluate! function removed - derivative functionality has been removed

# grad_type_dP function removed - derivative functionality has been removed


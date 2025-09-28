abstract type AbstractSChain{TT} end 
struct SChain{TT} <: AbstractSChain{TT}
   F::TT
end

struct TypedChain{TT, IN, OUT} <: AbstractSChain{TT}
   F::TT 
end 


# construct a chain recursively 
chain(F1, F2, args...) = chain( chain(F1, F2), args... )
# for most arguments, just form a tuple 
chain(F1, F2) = SChain( (F1, F2) )
# if one of them is a chain already, then combine into a single long chain 
chain(F1::SChain, F2) = SChain( tuple(F1.F..., F2) )
chain(F1, F2::SChain) = SChain( tuple(F1, F2.F...) )
chain(F1::SChain, F2::SChain) = chain( tuple(F1.F..., F2.F...) )

Base.length(c::SChain) = length(c.F)

@generated function evaluate(chain::AbstractSChain{TT}, X) where {TT}
   LEN = length(TT.types)
   code = Expr[]  
   push!(code, :(X_0 = X))
   for l = 1:LEN 
      push!(code, Meta.parse("F_$l = chain.F[$l]"))
      push!(code, Meta.parse("X_$l = evaluate(F_$l, X_$(l-1))"))
      push!(code, Meta.parse("release!(X_$(l-1))"))
   end
   push!(code, Meta.parse("return X_$LEN"))
   return Expr(:block, code...)
end

## 

import Base: == 

==(ch1::SChain, ch2::SChain) = 
      all( F1==F2 for (F1, F2) in zip(ch1.F, ch2.F) )

write_dict(chain::SChain) = Dict(
            "__id__" => "ACE_SChain", 
            "F" => write_dict.(chain.F)
         )

read_dict(::Val{:ACE_SChain}, D::Dict) = 
         SChain(tuple( read_dict.(D["F"])... ))


## ALTERNATIVE CHAIN IMPLEMENTATION - LINKS 

abstract type ChainLink end 

next(::ChainLink) = nothing 

previous(::ChainLink) = nothing 


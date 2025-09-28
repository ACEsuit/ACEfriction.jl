
# import ForwardDiff
import LegibleLambdas
import LegibleLambdas: LegibleLambda
import .ACE: read_dict, write_dict 

export λ, lambda 

struct Lambda{TL}
   ll::TL
   exstr::String
end

Base.show(io::IO, t::Lambda) = print(io, "λ($(t.exstr))")

function λ(str::String) 
   ex = Meta.parse(str)   
   ll = LegibleLambda(ex, eval(ex))
   return Lambda(ll, str)
end

lambda(str::String) = λ(str)

(t::Lambda)(x) = evaluate(t, x)

evaluate(t::Lambda, x) = t.ll.λ(x)


write_dict(t::Lambda)  = Dict(
         "__id__" => "ACE_Lambda", 
         "exstr" => t.exstr
      )

read_dict(::Val{:ACE_Lambda}, D::Dict) = λ(D["exstr"])

import Base: ==

==(F1::Lambda, F2::Lambda) = (F1.exstr == F2.exstr)


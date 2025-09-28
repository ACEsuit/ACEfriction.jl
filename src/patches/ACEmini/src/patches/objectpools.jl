

module ObjectPools

# This is inspired by 
#      https://github.com/tpapp/ObjectPools.jl
# but simplified and now evolved 

using DataStructures: Stack 

export acquire!, release!
export VectorPool, MatrixPool, ArrayPool
using Base.Threads: nthreads, threadid


# TODO: 
# * General ObjectPool 
# * convert to more general ArrayPool 
# * Consider allowing the most common "derived" types and enabling 
#   those via dispatch; e.g. Duals? Complex? All possible Floats? ....

struct VectorPool{T}
    #        tid    stack    object 
    arrays::Vector{Stack{Vector{T}}}
    VectorPool{T}() where {T} = new( [ Stack{Vector{T}}() for _=1:nthreads() ] )
end


acquire!(pool::VectorPool{T}, sz::Union{Integer, NTuple{N}}, ::Type{T}) where {T, N} = 
        acquire!(pool, sz)

acquire!(pool::VectorPool{T}, len::Integer, ::Type{T}) where {T} = 
        acquire!(pool, len)


function acquire!(pool::VectorPool{T}, len::Integer) where {T}
    tid = threadid()
    if !isempty(pool.arrays[tid]) > 0     
        x = pop!(pool.arrays[tid])
        if len != length(x)
            resize!(x, len)
        end
        return x 
    else
        return Vector{T}(undef, len)
    end
end

function release!(pool::VectorPool{T}, x::Vector{T}) where {T}
    tid = threadid() 
    push!(pool.arrays[tid], x)
    return nothing 
end

# Vector -> Array  -> Vector 

function acquire!(pool::VectorPool{T}, sz::NTuple{N}) where {T, N}
    tid = threadid()
    len = prod(sz)::Integer
    if !isempty(pool.arrays[tid])
        x = pop!(pool.arrays[tid])
        if len != length(x)
            resize!(x, len)
        end 
        return reshape(x, sz)
    else
        return Array{T, N}(undef, sz)
    end
end

release!(pool::VectorPool{T}, x::Array{T}) where {T} = 
        release!(pool, reshape(x, :))

# fallbacks that allocate and relase to the main GC

acquire!(pool::VectorPool{T}, len::Integer, S::Type{T1}) where {T, T1} = 
        Vector{S}(undef, len)

acquire!(pool::VectorPool{T}, sz::NTuple{N}, S::Type{T1}) where {T, N, T1} = 
        Array{T, N}(undef, sz)

release!(pool::VectorPool{T}, x::AbstractVector) where {T} = 
    nothing 


end # module


function vSimpleCTCinitδ(d, p, seq, TypeZERO)
    @inbounds if threadIdx().x == 1
        d[1,1] = culog(p[seq[1],1] + TypeZERO)
        d[2,1] = culog(p[seq[2],1] + TypeZERO)
    end
    return nothing
end

function vSimpleCTCfwd(d, ϕ, p, seq, L::Int, T::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    for t = 2:T
        τ = t-1
        first = CUDA.max(1, t-T+L-1)
        lasst = CUDA.min(1+t, L)
        @inbounds if first ≠ 1 # then each node has two kids
            for s = first : stride : lasst
                i = ifelse(d[s-1,τ] > d[s,τ], s-1, s)
                d[s,t] = d[i,τ] + culog(p[seq[s],t])
                ϕ[s,τ] = i
            end
        else
            d[first,t] = d[first,τ] + culog(p[blank,t])
            ϕ[first,τ] = 1
            for s = first+1 : stride : lasst
                i = ifelse(d[s-1,τ] > d[s,τ], s-1, s)
                d[s,t] = d[i,τ] + culog(p[seq[s],t])
                ϕ[s,τ] = i
            end
        end
        sync_threads()
    end
    return nothing
end

function vSimpleCTCend(h, d, L::Int, T::Int)
    @inbounds if threadIdx().x == 1
        h[T] = ifelse(d[L,T] > d[L-1,T], L, L-1)
    end
    return nothing
end

function vSimpleCTCbwd(h, ϕ, T::Int)
    if threadIdx().x == 1
        @inbounds for t = T-1:-1:1
            h[t] = ϕ[h[t+1],t]
        end
    end
    return nothing
end


function vSimpleCTCalign(r, h, p, seq, lnp, ONE::T) where T
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    @inbounds for t = start:stride:T
        i = seq[h[t]]
        r[i,t] = ONE
        CUDA.@atomic lnp[1] += log(p[i,t])
    end
    return nothing
end

"""
    ViterbiSimpleCTC(p::Array{F,2}, seqlabel::VecInt; blank::Int=1)
force alignment by viterbi algorithm

# Topology Example
     ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐
    ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐
    │blank├─►│  S  ├─►│blank├─►│  U  ├─►│blank├─►│  N  ├─►│blank│
    └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘
"""
function ViterbiSimpleCTC(p::CuArray{TYPE,2}, seqlabel::VecInt; blank::Int=1, eps::Real=1f-5) where TYPE
    S, T = size(p)                               # assert p is a 2-D tensor
    seq  = seqfastctc(seqlabel, blank)           # extend by topology constraint
    ZERO = TYPE(eps / (S-1))                     # typed value closing to 0 but bigger than 0
    ONE  = TYPE(1.0f0 - eps)                     # typed value closing to 1 but less than 1
    L    = length(seq)                           # topology length with blanks, assert L ≤ T
    lnp  = TYPE(0.0f0)

    if L == 1
        r = fill!(CuArray{TYPE,2}(undef,S,T), ZERO)
        r[blank,:] .= ONE
        return r, - sum(log.(p[blank,:]))
    end

    seq  = cu(seq)        # move to device
    Log0 = LogZero(TYPE)  # approximate -Inf of TYPE
    CUDA.@sync begin
        @async d = fill!(CuArray{TYPE,2}(undef,L,T), Log0)
        @async ϕ = CUDA.zeros(Int, L, T-1)
        @async h = CUDA.zeros(Int, T)
        @async lnp = fill!(CuArray{TYPE,1}(undef,1), ZERO);
    end

    # ══ init at fisrt timestep ══
    CUDA.@sync @cuda threads=1 vSimpleCTCinitδ(d, p, seq, ZERO)
    # ══ viterbi in log scale ══
    CUDA.@sync @cuda blocks=1 threads=CuThreads(L) vSimpleCTCfwd(d, ϕ, p, seq, L, T)

    # ══ backtrace ══
    CUDA.@sync @cuda threads=1 vSimpleCTCend(h, d, L, T)
    CUDA.@sync @cuda threads=1 vSimpleCTCbwd(h, ϕ, T)

    # ══ one-hot assignment ══
    CUDA.@sync @cuda threads=T vSimpleCTCalign(r, h, p, seq, lnp, ONE)

    return r, -Array(lnp)[1]
end

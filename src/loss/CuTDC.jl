function initTDCio(a, b, p, seq, L::Int, T::Int, TypeZero)
    @inbounds if threadIdx().x == 1
        a[1,1] = culog(p[seq[1],1])
        a[2,1] = culog(p[seq[2],1])
        b[L-1,T] = TypeZero
        b[L,  T] = TypeZero
    end
    return nothing
end


# -- tdc-forward CUDA kernel
function tdcfwd(a, p, seq, L::Int, T::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    for t = 2:T
        τ = t-1
        @inbounds for s = start:stride:L
            if s≠1
                R = mod(s,4)
                if R==3 || R==0 || s==2
                    a[s,t] = CuLogSum2Exp(a[s,τ], a[s-1,τ])
                elseif R==2
                    a[s,t] = CuLogSum4Exp(a[s,τ], a[s-1,τ], a[s-2,τ], a[s-3,τ])
                elseif R==1
                    a[s,t] = a[s-1,τ]
                end
            else
                a[s,t] = a[s,τ]
            end
            a[s,t] += culog(p[seq[s],t])
        end
        sync_threads()
    end
    return nothing
end


# -- tdc-backward CUDA kernel
function tdcbwd(b, p, seq, L::Int, T::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    for t = T-1:-1:1
        τ = t+1
        @inbounds for s = start:stride:L
            Q⁰ = b[s,τ] + culog(p[seq[s],τ])
            if s≠L
                R = mod(s,4)
                Q¹ = b[s+1,τ] + culog(p[seq[s+1],τ])
                if R==2 || s==1 || s==L-1
                    b[s,t] = CuLogSum2Exp(Q⁰, Q¹)
                elseif R==0
                    Q² = b[s+2,τ] + culog(p[seq[s+2],τ])
                    b[s,t] = CuLogSum3Exp(Q⁰, Q¹, Q²)
                elseif R==3
                    Q³ = b[s+3,τ] + culog(p[seq[s+3],τ])
                    b[s,t] = CuLogSum3Exp(Q⁰, Q¹, Q³)
                elseif R==1
                    b[s,t] = Q¹
                end
            else
                b[s,t] = Q⁰
            end
        end
        sync_threads()
    end
    return nothing
end


# -- LogLikely of tdc CUDA kernel
function tdclogsum(a, b, logsum)
    if threadIdx().x == 1
        logsum[1] = CuLogSum2Exp(a[1,1] + b[1,1], a[2,1] + b[2,1])
    end
    return nothing
end


# -- γ = α + β CUDA kernel
function tdcgamma(g, a, b, logsum, N::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    for i = start:stride:N
        @inbounds g[i] = cuexp(a[i] + b[i] - logsum[1])
    end
    return nothing
end


# -- reduce lines of γ
function TDCReduce(r, g, seq, N::Int, T::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    for n = 1:N
        s = n<<2
        @inbounds for t = start:stride:T
            r[seq[s-2],t] += g[s-2,t]            # reduce front states
            r[seq[s-1],t] += g[s-1,t]            # reduce labels' states
            r[seq[s  ],t] += g[s-3,t] + g[s,t]   # reduce blank states
        end
        sync_threads()
    end
    return nothing
end


"""
    TDC(p::CuArray{T,2}, seqlabel::Vector{Int}; blank::Int=1, front::Int=2) where T

# Topology Example
       1        2        3        4            1        2        3        4            1        2        3        4
     ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐                 ┌─►─┐    ┌─►─┐    ┌─►─┐                 ┌─►─┐    ┌─►─┐    ┌─►─┐
    ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐      ┌─────┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐      ┌─────┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐
    │blank├─►│front├─►│  A  ├─►│blank├──•──►│blank├─►│front├─►│  B  ├─►│blank├──•──►│blank├─►│front├─►│  C  ├─►│blank│
    └─────┘  └─────┘  └──┬──┘  └──┬──┘      └─────┘  └┬───┬┘  └──┬──┘  └──┬──┘      └─────┘  └┬───┬┘  └─────┘  └─────┘
                         │        └───────────►───────┘   │      │        └───────────►───────┘   │
                         └────────────────────►───────────┘      └────────────────────►───────────┘

"""
function Mira.TDC(p::CuArray{TYPE,2}, seqlabel::Vector{Int}; blank::Int=1, front::Int=2) where TYPE
    seq  = seqtdc(seqlabel, blank, front)
    Log0 = LogZero(TYPE)
    ZERO = TYPE(0)
    S, T = size(p)
    L = length(seq)
    G = L * T

    if L == 1
        r = fill!(CuArray{TYPE,2}(undef,S,T), ZERO)
        r[blank,:] .= TYPE(1)
        return r, - sum(log.(p[blank,:]))
    end

    seq = cu(seq)
    CUDA.@sync begin
        @async a = fill!(CuArray{TYPE,2}(undef,L,T), Log0);
        @async b = fill!(CuArray{TYPE,2}(undef,L,T), Log0);
        @async g = fill!(CuArray{TYPE,2}(undef,L,T), ZERO);
        @async r = fill!(CuArray{TYPE,2}(undef,S,T), ZERO);
        @async LOGSUM = fill!(CuArray{TYPE,1}(undef,1), Log0);
    end

    CUDA.@sync @cuda threads=1 initTDCio(a, b, p, seq, L, T, ZERO);

    CUDA.@sync begin
        @async @cuda blocks=1 threads=CuThreads(L) tdcfwd(a, p, seq, L, T);
        @async @cuda blocks=1 threads=CuThreads(L) tdcbwd(b, p, seq, L, T);
    end

    CUDA.@sync @cuda                    threads=1            tdclogsum(a, b, LOGSUM);
    CUDA.@sync @cuda blocks=CuBlocks(G) threads=CuThreads(G) tdcgamma(g, a, b, LOGSUM, G);
    CUDA.@sync @cuda blocks=CuBlocks(T) threads=CuThreads(T) TDCReduce(r, g, seq, div(L,4), T);

    return r, -Array(LOGSUM)[1]
end

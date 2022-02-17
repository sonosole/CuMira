function initFastCTCio(a, b, p, seq, L::Int, T::Int, TypeZero)
    if threadIdx().x == 1
        a[1,1] = culog(p[seq[1],1])
        a[2,1] = culog(p[seq[2],1])
        b[L-1,T] = TypeZero
        b[L,  T] = TypeZero
    end
    return nothing
end


# -- fastctc-forward CUDA kernel
function fastctcfwd(a, p, seq, L::Int, T::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    for t = 2:T
        τ = t-1
        first = CUDA.max(1, L-2*(T-t)-1)
        lasst = CUDA.min(2*t, L)
        for s = start:stride:L
            if first <= s <= lasst
                if s≠1
                    a[s,t] = CuLogSum2Exp(a[s,τ], a[s-1,τ])
                else
                    a[s,t] = a[s,τ]
                end
                a[s,t] += culog(p[seq[s],t])
            end
        end
        sync_threads()
    end
    return nothing
end


# -- fastctc-backward CUDA kernel
function fastctcbwd(b, p, seq, L::Int, T::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    for t = T-1:-1:1
        τ = t+1
        first = CUDA.max(1, L-2*(T-t)-1)
        lasst = CUDA.min(2*t, L)
        for s = start:stride:L
            if first <= s <= lasst
                Q = b[s,τ] + log(p[seq[s],τ])
                if s≠L
                    b[s,t] = LogSum2Exp(Q, b[s+1,τ] + log(p[seq[s+1],τ]))
                else
                    b[s,t] = Q
                end
            end
        end
        sync_threads()
    end
    return nothing
end


# -- LogLikely of fastctc CUDA kernel
function fastctclogsum(a, b, logsum)
    if threadIdx().x == 1
        logsum[1] = CuLogSum2Exp(a[1,1] + b[1,1], a[2,1] + b[2,1])
    end
    return nothing
end


# -- γ = α + β   CUDA kernel
function fastctcgamma(g, a, b, logsum, N::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    for i = start:stride:N
        @inbounds g[i] = cuexp(a[i] + b[i] - logsum[1])
    end
    return nothing
end


# -- reduce first line of γ, i.e. blank
function FastCTCReduceFirst(r, g, T::Int, blank::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    for t = start:stride:T
        @inbounds r[blank,t] = g[1,t]
    end
    return nothing
end


# -- reduce rest lines of γ, i.e. non-blank
function FastCTCReduceOther(r, g, seq, N::Int, T::Int, blank::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    for n = 1:N
        s = n<<1
        for t = start:stride:T
            @inbounds r[seq[s],t] += g[s,  t]  # reduce labels' states
            @inbounds r[blank, t] += g[s+1,t]  # reduce blank state
        end
        sync_threads()
    end
    return nothing
end


# CUDA version of CTC LOSS
function Mira.FastCTC(p::CuArray{TYPE,2}, seqlabel; blank::Int=1) where TYPE
    seq  = seqfastctc(seqlabel, blank)
    Log0 = LogZero(TYPE)
    ZERO = TYPE(0)
    S, T = size(p)
    L = length(seq)        # topology length with blanks
    G = L * T

    if L == 1
        r = fill!(CuArray{TYPE,2}(undef,S,T), ZERO);
        r[blank,:] .= TYPE(1)
        return r, - sum(log.(p[blank,:]))
    end

    seq = cu(seq)
    CUDA.@sync begin
        a = fill!(CuArray{TYPE,2}(undef,L,T), Log0);
        b = fill!(CuArray{TYPE,2}(undef,L,T), Log0);
        g = fill!(CuArray{TYPE,2}(undef,L,T), ZERO);
        r = fill!(CuArray{TYPE,2}(undef,S,T), ZERO);
        LOGSUM = fill!(CuArray{TYPE,1}(undef,1), Log0);
    end

    CUDA.@sync @cuda threads=1 initFastCTCio(a, b, p, seq, L, T, ZERO);

    CUDA.@sync begin
        @cuda blocks=1 threads=CuThreads(L) fastctcfwd(a, p, seq, L, T);
        @cuda blocks=1 threads=CuThreads(L) fastctcbwd(b, p, seq, L, T);
    end

    CUDA.@sync @cuda                    threads=1            fastctclogsum(a, b, LOGSUM);
    CUDA.@sync @cuda blocks=CuBlocks(G) threads=CuThreads(G) fastctcgamma(g, a, b, LOGSUM, G);
    CUDA.@sync @cuda blocks=CuBlocks(T) threads=CuThreads(T) FastCTCReduceFirst(r, g, T, blank);
    CUDA.@sync @cuda blocks=CuBlocks(T) threads=CuThreads(T) FastCTCReduceOther(r, g, seq, div(L-1,2), T, blank);

    return r,-Array(LOGSUM)[1]
end

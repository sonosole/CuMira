# -- ctc-forward CUDA kernel
function ctcfwd(a, p, seq, L::Int, T::Int, blank::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    for t = 2:T
        τ = t-1
        first = CUDA.max(1,L-2*(T-t)-1);
        lasst = CUDA.min(2*t,L);
        for s = start:stride:L
            if first <= s <= lasst
                i = div(s,2);
                if s==1
                    a[s,t] = a[s,τ] + culog(p[blank,t])
                elseif mod(s,2)==1
                    a[s,t] = CuLogSum2Exp(a[s,τ], a[s-1,τ]) + culog(p[blank,t])
                elseif s==2
                    a[s,t] = CuLogSum2Exp(a[s,τ], a[s-1,τ]) + culog(p[seq[i],t])
                elseif seq[i]==seq[i-1]
                    a[s,t] = CuLogSum2Exp(a[s,τ], a[s-1,τ]) + culog(p[seq[i],t])
                else
                    a[s,t] = CuLogSum3Exp(a[s,τ], a[s-1,τ], a[s-2,τ]) + culog(p[seq[i],t])
                end
            end
        end
        sync_threads()
    end
    return nothing
end

# -- ctc-backward CUDA kernel
function ctcbwd(b, p, seq, L::Int, T::Int, blank::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    for t = T-1:-1:1
        τ = t+1
        first = CUDA.max(1,L-2*(T-t)-1)
        lasst = CUDA.min(2*t,L)
        for s = start:stride:L
            if first <= s <= lasst
                i = div(s,2)
                j = div(s+1,2)
                if s==L
                    b[s,t] = b[s,τ] + culog(p[blank,τ])
                elseif mod(s,2)==1
                    b[s,t] = CuLogSum2Exp(b[s,τ] + culog(p[blank, τ]), b[s+1,τ] + culog(p[seq[j],τ]))
                elseif s==L-1
                    b[s,t] = CuLogSum2Exp(b[s,τ] + culog(p[seq[i],τ]), b[s+1,τ] + culog(p[blank,τ]))
                elseif seq[i]==seq[i+1]
                    b[s,t] = CuLogSum2Exp(b[s,τ] + culog(p[seq[i],τ]), b[s+1,τ] + culog(p[blank,τ]))
                else
                    b[s,t] = CuLogSum3Exp(b[s,τ] + culog(p[seq[i],τ]), b[s+1,τ] + culog(p[blank,τ]), b[s+2,τ] + culog(p[seq[i+1],τ]))
                end
            end
        end
        sync_threads()
    end
    return nothing
end

# -- ctc-initital CUDA kernel
function initCTCio(a, b, p, seq, L::Int, T::Int, TypeZero, blank::Int)
    if threadIdx().x == 1
        a[1,1] = culog(p[blank,1])
        a[2,1] = culog(p[seq[1],1])
        b[L  ,T] = TypeZero
        b[L-1,T] = TypeZero
    end
    return nothing
end

# -- LogLikely of ctc CUDA kernel
function ctclogsum(a, b, logsum)
    if threadIdx().x == 1
        logsum[1] = CuLogSum2Exp(a[1,1] + b[1,1], a[2,1] + b[2,1])
    end
    return nothing
end

# -- γ = α + β   CUDA kernel
function ctcgamma(g, a, b, logsum, N)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    for i = start:stride:N
        @inbounds g[i] = cuexp(a[i] + b[i] - logsum[1])
    end
    return nothing
end

# -- reduce first line of γ, i.e. blank
function CTCReduceFirst(r, g, T::Int, blank::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    for t = start:stride:T
        @inbounds r[blank,t] = g[1,t]
    end
    return nothing
end

# -- reduce rest lines of γ, i.e. non-blank
function CTCReduceOther(r, g, seq, N::Int, T::Int, blank::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    for n = 1:N
        s = n<<1
        for t = start:stride:T
            @inbounds r[seq[n],t] += g[s,  t]  # reduce labels' states
            @inbounds r[blank, t] += g[s+1,t]  # reduce blank state
        end
        sync_threads()
    end
    return nothing
end

# CUDA version of CTC LOSS
function Mira.CTC(p::CuArray{TYPE,2}, seq::Vector{Int}; blank::Int=1) where TYPE
    ZERO = TYPE(0)
    S, T = size(p)

    if seq[1] == 0
        r = fill!(CuArray{TYPE,2}(undef,S,T), ZERO)
        r[blank,:] .= TYPE(1)
        return r, - sum(log.(p[blank,:]))
    end

    L = 2 * length(seq) + 1
    G = L * T
    seq  = cu(seq)
    Log0 = LogZero(TYPE)
    CUDA.@sync begin
        a = fill!(CuArray{TYPE,2}(undef,L,T), Log0);
        b = fill!(CuArray{TYPE,2}(undef,L,T), Log0);
        g = fill!(CuArray{TYPE,2}(undef,L,T), ZERO);
        r = fill!(CuArray{TYPE,2}(undef,S,T), ZERO);
        LOGSUM = fill!(CuArray{TYPE,1}(undef,1), Log0);
    end

    CUDA.@sync @cuda threads=1 initCTCio(a, b, p, seq, L, T, ZERO, blank);

    CUDA.@sync begin
        @cuda blocks=1 threads=CuThreads(L) ctcfwd(a, p, seq, L, T, blank);
        @cuda blocks=1 threads=CuThreads(L) ctcbwd(b, p, seq, L, T, blank);
    end

    CUDA.@sync @cuda                    threads=1            ctclogsum(a, b, LOGSUM);
    CUDA.@sync @cuda blocks=CuBlocks(G) threads=CuThreads(G) ctcgamma(g, a, b, LOGSUM, G);
    CUDA.@sync @cuda blocks=CuBlocks(T) threads=CuThreads(T) CTCReduceFirst(r, g, T, blank);
    CUDA.@sync @cuda blocks=CuBlocks(T) threads=CuThreads(T) CTCReduceOther(r, g, seq, length(seq), T, blank);

    return r,-Array(LOGSUM)[1]
end

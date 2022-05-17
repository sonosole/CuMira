# -- tcs-initital CUDA kernel
function initTCSio(a, b, p, seq, L::Int, T::Int, TypeZero)
    if threadIdx().x == 1
    	a[1,1] = culog(p[seq[1],1])
        a[2,1] = culog(p[seq[2],1])
        b[L-1,T] = TypeZero
    	b[L,  T] = TypeZero
    end
    return nothing
end

# -- tcs-forward CUDA kernel
function tcsfwd(a, p, seq, L::Int, T::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    for t = 2:T
        τ = t-1
        for s = start:stride:L
            if s!=1
                R = mod(s,3)
                if R==1 || s==2 || R==0
                    a[s,t] = CuLogSum2Exp(a[s,τ], a[s-1,τ])
                elseif R==2
                    a[s,t] = CuLogSum3Exp(a[s,τ], a[s-1,τ], a[s-2,τ])
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

# -- tcs-backward CUDA kernel
function tcsbwd(b, p, seq, L::Int, T::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    for t = T-1:-1:1
        τ = t+1
        for s = start:stride:L
            Q = b[s,τ] + culog(p[seq[s],τ])
            if s!=L
                R = mod(s,3)
                V = b[s+1,τ] + culog(p[seq[s+1],τ])
                if R==1 || R==2 || s==L-1
                    b[s,t] = CuLogSum2Exp(Q, V)
                elseif R==0
                    b[s,t] = CuLogSum3Exp(Q, V, b[s+2,τ] + culog(p[seq[s+2],τ]))
                end
            else
                b[s,t] = Q
            end
        end
        sync_threads()
    end
    return nothing
end

# -- LogLikely of tcs CUDA kernel
function tcslogsum(a, b, logsum)
    if threadIdx().x == 1
        logsum[1] = CuLogSum2Exp(a[1,1] + b[1,1], a[2,1] + b[2,1])
    end
    return nothing
end

# -- γ = α + β   CUDA kernel
function tcsgamma(g, a, b, logsum, N::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    for i = start:stride:N
        @inbounds g[i] = cuexp(a[i] + b[i] - logsum[1])
    end
    return nothing
end

# -- reduce first line of γ
function TCSReduceFirst(r, g, seq, T::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    for t = start:stride:T
        @inbounds r[seq[1],t] = g[1,t]
    end
    return nothing
end

# -- reduce rest lines of γ
function TCSReduceOther(r, g, seq, N::Int, T::Int)
    stride = blockDim().x * gridDim().x
    start  = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    for n = 1:N
        s = 3*n
        for t = start:stride:T
            @inbounds r[seq[s-1],t] += g[s-1,t]  # reduce forground state
            @inbounds r[seq[s  ],t] += g[s,  t]  # reduce labels' states
            @inbounds r[seq[s+1],t] += g[s+1,t]  # reduce background state
        end
        sync_threads()
    end
    return nothing
end


function Mira.TCS(p::CuArray{TYPE,2}, seqlabel::Vector{Int}; background::Int=1, foreground::Int=2) where TYPE
    seq  = seqtcs(seqlabel, background, foreground)
    ZERO = TYPE(0)
    S, T = size(p)
    L = length(seq)
    G = L * T

    if L == 1
        r = fill!(CuArray{TYPE,2}(undef,S,T), ZERO);
        r[background,:] .= TYPE(1)
        return r, - sum(log.(p[background,:]))
    end

    seq  = cu(seq)
    Log0 = LogZero(TYPE)
    CUDA.@sync begin
        a = fill!(CuArray{TYPE,2}(undef,L,T), Log0);
        b = fill!(CuArray{TYPE,2}(undef,L,T), Log0);
        g = fill!(CuArray{TYPE,2}(undef,L,T), ZERO);
        r = fill!(CuArray{TYPE,2}(undef,S,T), ZERO);
        LOGSUM = fill!(CuArray{TYPE,1}(undef,1), Log0);
    end

    CUDA.@sync @cuda threads=1 initTCSio(a, b, p, seq, L, T, ZERO);

    CUDA.@sync begin
        @cuda blocks=1 threads=CuThreads(L) tcsfwd(a, p, seq, L, T);
        @cuda blocks=1 threads=CuThreads(L) tcsbwd(b, p, seq, L, T);
    end

    CUDA.@sync @cuda                    threads=1            tcslogsum(a, b, LOGSUM);
    CUDA.@sync @cuda blocks=CuBlocks(G) threads=CuThreads(G) tcsgamma(g, a, b, LOGSUM, G);
    CUDA.@sync @cuda blocks=CuBlocks(T) threads=CuThreads(T) TCSReduceFirst(r, g, seq, T);
    CUDA.@sync @cuda blocks=CuBlocks(T) threads=CuThreads(T) TCSReduceOther(r, g, seq, div(L-1,3), T);

    return r, -Array(LOGSUM)[1]
end

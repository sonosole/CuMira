export DataParallelX

"""
mutable struct DataParallelX{T} <: Parallel

# Constructor
    DataParallelX{T}(model     :: T;
                    master    :: Int=0,            # master device
                    devices   :: Vector{Int}=[0],  # workers devices
                    criterion :: Function,         # loss function
                    xspliter  :: Spliter,          # split input flags for devices
                    yspliter  :: Spliter,          # split label flags for devices
                    type      :: Type=CuArray{Float32}) where T
"""
mutable struct DataParallelX{T} <: Parallel
    masteridx :: Int                      # master device's index
    devices   :: Vector{Int}              # workers devices
    criterion :: Function                 # loss function
    xspliter  :: Spliter                  # split input flags
    yspliter  :: Spliter                  # split label flags
    models    :: Vector{T}                # models on `devices`
    params    :: Vector{Vector{Variable}} # workers-device's params
    caches    :: Vector                   # caches on master `devices`
    tuples    :: Vector                   # parallel index pairs
    type      :: Type
    function DataParallelX{T}(model     :: T;
                             master    :: Int=0,
                             devices   :: Vector{Int}=[0],
                             criterion :: Function,
                             xspliter  :: Spliter=(dim=1, keptsame=false),
                             yspliter  :: Spliter=(dim=1, keptsame=false),
                             type      :: Type=CuArray{Float32}) where T

        @assert master in devices "master=$master not in devices=$devices"
        masteridx = 0
        caches = Vector()
        tuples = Vector()
        ntasks = length(devices)
        models = Vector{T}(undef, ntasks)
        params = Vector{Vector{Variable}}(undef, ntasks)
        for i  = 1:ntasks
            device!(devices[i])
            models[i] = clone(model, type=type)
            params[i] = paramsof(models[i])
            if devices[i] == master
                masteridx = i
                for x in params[i]
                    push!(caches, zero(ᵛ(x)))
                end
            else
                for j = 1:length(params[i])
                    push!(tuples, (i,j))
                end
            end
        end
        device!(devices[masteridx])
        new{T}(masteridx,
               devices,
               criterion,
               xspliter,
               yspliter,
               models,
               params,
               caches,
               tuples,
               type)
    end
end


function DataParallelX(model     :: T;
                      master    :: Int=0,
                      devices   :: Vector{Int}=[0],
                      criterion :: Function,
                      xspliter  :: Spliter=(dim=1, keptsame=false),
                      yspliter  :: Spliter=(dim=1, keptsame=false),
                      type      :: Type=CuArray{Float32}) where T

    return DataParallelX{T}(model;
                           master    = master,
                           devices   = devices,
                           criterion = criterion,
                           xspliter  = xspliter,
                           yspliter  = yspliter,
                           type      = type)
end


function Base.show(io::IO, dp::DataParallelX{T}) where T
    println("DataParallelX{$T}")
    println(io, "——————————————————————————————————————————————")
    println(io, "master device  = $(dp.masteridx)")
    println(io, "worker devices = $(dp.devices)")
    println(io, "     criterion = $(dp.criterion)")
    println(io, "      xspliter = $(dp.xspliter)")
    println(io, "      yspliter = $(dp.yspliter)")
    println(io, "          type = $(dp.type)")
    println(io, "——————————————————————————————————————————————")
end


masterof(dp::DataParallelX)  = dp.models[dp.masteridx]

function Mira.xparamsof(dp::DataParallelX)
    device!(dp.devices[dp.masteridx])
    return xparamsof(masterof(dp))
end

function masterdevice!(dp::DataParallelX)
    device!(dp.devices[dp.masteridx])
    return nothing
end


function fwdbwd(dp::DataParallelX, x, y)
    T = dp.type
    G = dp.params
    M = dp.masteridx
    C = length(G[M])        # number of learnable params
    D = length(dp.devices)  # number of GPUs
    l = Vector(undef,   D)  # to store losses
    caches = dp.caches
    xdim, xkeptsame = dp.xspliter
    ydim, ykeptsame = dp.yspliter
    I₁, I₂, N = keptdims(x; dim=xdim)
    J₁, J₂, L = keptdims(y; dim=ydim)
    @assert N == L "#input=$N and #label=$L do NOT have the same number of samples"
    batchsize = ceil(Int, N/D)

    # forward, loss and backward
    @sync begin
        Threads.@threads for i = 1:D
            device!(dp.devices[i])
            k = getidx(N, batchsize, i)
            input = xkeptsame ? x[I₁,k,I₂] : Variable{T}(x[I₁,k,I₂], true, false, true)
            label = ykeptsame ? y[J₁,k,J₂] : Variable{T}(y[J₁,k,J₂], true, false, true)
            v = forward(dp.models[i], input)
            c = dp.criterion(v,       label)
            backward(c)
            l[i] = cost(c)
        end
    end

    # reduce gradients and zero gradients of non-master's
    @sync begin
        for i = 1:D
            if i ≠ M
                device!(dp.devices[M])
                Threads.@threads for j = 1:C
                    copyto!(caches[j], δ(G[i][j]))
                    δ(G[M][j]) .+= caches[j]
                end
                device!(dp.devices[i])
                zerograds!(G[i])
            end
        end
    end
    device!(dp.devices[M])
    return sum(l)
end


function sync(dp::DataParallelX)
    T = dp.type
    G = dp.params
    M = dp.masteridx
    C = length(G[M])        # number of learnable params
    D = length(dp.devices)  # number of GPUs

    # move weights from master-GPU to non-master-GPUs
    @sync begin
        Threads.@threads for (i, j) in dp.tuples
            device!(dp.devices[i])
            copyto!(ᵛ(G[i][j]), ᵛ(G[M][j]))
        end
    end
    device!(dp.devices[M])
    return nothing
end

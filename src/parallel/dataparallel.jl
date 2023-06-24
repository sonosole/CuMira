export DataParallel
export masterdevice!
export masterof
export fwdbwd
export Spliter
export sync

const Spliter = NamedTuple{(:dim,:keptsame),Tuple{Int, Bool}}

"""
mutable struct DataParallel{T} <: Parallel

# Constructor
    DataParallel{T}(model     :: T;
                    master    :: Int=0,            # master device
                    devices   :: Vector{Int}=[0],  # workers devices
                    criterion :: Function,         # loss function
                    xspliter  :: Spliter,          # split input flags for devices
                    yspliter  :: Spliter,          # split label flags for devices
                    type      :: Type=CuArray{Float32}) where T
"""
mutable struct DataParallel{T} <: Parallel
    masteridx :: Int                      # master device's index
    devices   :: Vector{Int}              # workers devices
    criterion :: Function                 # loss function
    xspliter  :: Spliter                  # split input flags
    yspliter  :: Spliter                  # split label flags
    models    :: Vector{T}                # models on `devices`
    params    :: Vector{Vector{Variable}} # workers-device's params
    tuples    :: Vector{Tuple{Int,Int}}   # parallel index pairs
    type      :: Type
    function DataParallel{T}(model     :: T;
                             master    :: Int=0,
                             devices   :: Vector{Int}=[0],
                             criterion :: Function,
                             xspliter  :: Spliter=(dim=1, keptsame=false),
                             yspliter  :: Spliter=(dim=1, keptsame=false),
                             type      :: Type=CuArray{Float32}) where T

        @assert master in devices "master=$master not in devices=$devices"
        masteridx = 0
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
               tuples,
               type)
    end
end


function DataParallel(model     :: T;
                      master    :: Int=0,
                      devices   :: Vector{Int}=[0],
                      criterion :: Function,
                      xspliter  :: Spliter=(dim=1, keptsame=false),
                      yspliter  :: Spliter=(dim=1, keptsame=false),
                      type      :: Type=CuArray{Float32}) where T

    return DataParallel{T}(model;
                           master    = master,
                           devices   = devices,
                           criterion = criterion,
                           xspliter  = xspliter,
                           yspliter  = yspliter,
                           type      = type)
end


function Base.show(io::IO, dp::DataParallel{T}) where T
    print(io,   "═════════════════════════════════════════════")
    println("\n DataParallel{$T}")
    println(io, "═════════════════════════════════════════════")
    println(io, " master device  = $(dp.devices[dp.masteridx])")
    println(io, " worker devices = $(dp.devices)")
    println(io, "      criterion = $(dp.criterion)")
    println(io, "       xspliter = $(dp.xspliter)")
    println(io, "       yspliter = $(dp.yspliter)")
    println(io, "          dtype = $(eltype(dp.type))")
    println(io, "═════════════════════════════════════════════")
end


masterof(dp::DataParallel)  = dp.models[dp.masteridx]

function Mira.xparamsof(dp::DataParallel)
    device!(dp.devices[dp.masteridx])
    return xparamsof(masterof(dp))
end

function masterdevice!(dp::DataParallel)
    device!(dp.devices[dp.masteridx])
    return nothing
end


function fwdbwd(dp::DataParallel, x, y)
    T = dp.type
    G = dp.params
    M = dp.masteridx
    C = length(G[M])        # number of params
    D = length(dp.devices)  # number of GPUs
    l = Vector(undef,   D)  # to store losses
    xdim, xkeptsame = dp.xspliter
    ydim, ykeptsame = dp.yspliter
    I₁, I₂, N = keptdims(x; dim=xdim)
    J₁, J₂, L = keptdims(y; dim=ydim)
    @assert N == L "#input=$N and #label=$L do NOT have the same number of samples"
    batchsize = ceil(Int, N/D)

    # forward, loss and backward
    @sync for i = 1:D
        @async begin
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

    for i = 1:D
        if i ≠ M
            # reduce gradients
            @sync for j = 1:C
                @async begin
                    device!(dp.devices[M])
                    tmp = Zeros(T, G[M][j].shape)
                    copyto!(tmp, δ(G[i][j]))
                    δ(G[M][j]) .+= tmp
                end
            end
            # and zero gradients of non-master's
            @sync begin
                device!(dp.devices[i])
                zerograds!(G[i])
            end
        end
    end
    device!(dp.devices[M])
    return sum(l)/D
end


function sync(dp::DataParallel)
    T = dp.type
    G = dp.params
    M = dp.masteridx
    C = length(G[M])        # number of params
    D = length(dp.devices)  # number of GPUs

    # move weights from master-GPU to non-master-GPUs
    @sync for (i, j) in dp.tuples
        @async begin
            device!(dp.devices[i])
            copyto!(ᵛ(G[i][j]), ᵛ(G[M][j]))
        end
    end
    device!(dp.devices[M])
    return nothing
end

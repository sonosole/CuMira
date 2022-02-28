# this implementation is under the condition that:
# Cross-device copy of wrapped arrays would fail

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
    cpuvars   :: Vector{Variable}         # params on CPU
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
        cpuvars = paramsof(model)
        ntasks  = length(devices)
        models  = Vector{T}(undef, ntasks)
        params  = Vector{Vector{Variable}}(undef, ntasks)
        for i = 1:ntasks
            if devices[i] == master
                masteridx = i
            end
            device!(devices[i])
            models[i] = clone(model, type=type)
            params[i] = paramsof(models[i])
        end

        device!(devices[masteridx])

        for v in cpuvars
            zerodelta(v)
        end

        new{T}(masteridx,
               devices,
               criterion,
               xspliter,
               yspliter,
               models,
               params,
               cpuvars,
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
    println("DataParallel{$T}")
    println(io, "——————————————————————————————————————————————")
    println(io, "master device  = $(dp.masteridx)")
    println(io, "worker devices = $(dp.devices)")
    println(io, "     criterion = $(dp.criterion)")
    println(io, "      xspliter = $(dp.xspliter)")
    println(io, "      yspliter = $(dp.yspliter)")
    println(io, "          type = $(dp.type)")
    println(io, "——————————————————————————————————————————————")
end


masterof(dp::DataParallel) = dp.models[dp.masteridx]

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
    C = length(dp.cpuvars)
    D = length(dp.devices)
    l = Vector(undef,   D)
    xdim, xkeptsame = dp.xspliter
    ydim, ykeptsame = dp.yspliter
    I₁, I₂, N = keptdims(x; dim=xdim)
    J₁, J₂, L = keptdims(y; dim=ydim)
    @assert N == L "#input=$N and #label=$L do NOT have the same number of samples"
    batchsize = ceil(Int, N/D)

    # forward and loss and backward
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

    # reduce gradients from non-master-GPUs to CPU
    @sync begin
        for i = 1:D
            if i ≠ M
                device!(dp.devices[i])
                Threads.@threads for j = 1:C
                    δ(dp.cpuvars[j]) .+= Array(δ(G[i][j]))
                end
            end
        end
    end
    # zero gradients of non-master-GPUs
    @sync begin
        Threads.@threads for i = 1:D
            if i ≠ M
                device!(dp.devices[i])
                zerograds!(G[i])
            end
        end
    end
    # move gradients from CPU to master-GPU
    # and reset CPU's gradients to zero
    device!(dp.devices[M])
    Threads.@threads for j = 1:C
        δ(G[M][j]) .+= T(δ(dp.cpuvars[j]))
        δ(dp.cpuvars[j]) .= 0.0f0
    end
    return sum(l)
end


function sync(dp::DataParallel)
    T = dp.type
    G = dp.params
    M = dp.masteridx
    C = length(dp.cpuvars)
    D = length(dp.devices)

    # move weights from master-GPU to CPU
    device!(dp.devices[M])
    Threads.@threads for j = 1:C
        ᵛ(dp.cpuvars[j]) .= Array(ᵛ(G[M][j]))
    end

    # copy weights from CPU to non-master-GPUs
    @sync begin
        for i = 1:D
            if i ≠ M
                device!(dp.devices[i])
                Threads.@threads for j = 1:C
                    copyto!(ᵛ(G[i][j]), T(ᵛ(dp.cpuvars[j])))
                end
            end
        end
    end
    device!(dp.devices[M])
    return nothing
end

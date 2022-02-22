export DataParallel
export masterof
export fwdbwd

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
                    type      :: Type=Array{Float32}) where T
"""
mutable struct DataParallel{T} <: Parallel
    masteridx :: Int                      # master device's index
    devices   :: Vector{Int}              # workers devices
    criterion :: Function                 # loss function
    xspliter  :: Spliter                  # split input flags
    yspliter  :: Spliter                  # split label flags
    models    :: Vector{T}                # models on `devices`
    params    :: Vector{Vector{Variable}} # workers-device's params
    type      :: Type
    function DataParallel{T}(model     :: T;
                             master    :: Int=0,
                             devices   :: Vector{Int}=[0],
                             criterion :: Function,
                             xspliter  :: Spliter=(dim=1, keptsame=false),
                             yspliter  :: Spliter=(dim=1, keptsame=false),
                             type      :: Type=Array{Float32}) where T

        @assert master in devices "master=$master not in devices=$devices"
        masteridx = -1
        ntasks = length(devices)
        models = Vector{T}(undef, ntasks)
        params = Vector{Vector{Variable}}(undef, ntasks)
        for i  = 1:ntasks
            if devices[i] == master
                masteridx = i
            end
            device!(devices[i])
            models[i] = clone(model, type=type)
            params[i] = paramsof(models[i])
        end

        new{T}(masteridx,
               devices,
               criterion,
               xspliter,
               yspliter,
               models,
               params,
               type)
    end
end


function Base.show(io, dp::DataParallel{T}) where T
    println("DataParallel{$T}")
    println(io, "———————————————————————")
    println(io, "master device  = $(dp.masteridx)")
    println(io, "worker devices = $(dp.devices)")
    println(io, "     criterion = $(dp.criterion)")
    println(io, "      xspliter = $(dp.xspliter)")
    println(io, "      yspliter = $(dp.yspliter)")
    println(io, "          type = $(dp.type)")
    println(io, "———————————————————————")
end


masterof(dp::DataParallel)  = dp.models[dp.masteridx]
Mira.xparamsof(dp::DataParallel) = xparamsof(masterof(dp))


function fwdbwd(dp::DataParallel, x, y)
    T = dp.type
    n = length(dp.devices)
    l = Vector(undef,   n)
    xdim, xkeptsame = dp.xspliter
    ydim, ykeptsame = dp.yspliter
    I₁, I₂, N = keptdims(x; dim=xdim)
    J₁, J₂, M = keptdims(y; dim=ydim);
    @assert N == M "input and label do NOT have the same samples"
    batchsize = div(N, n)

    # forward and loss and backward
    @sync begin
        Threads.@threads for i = 1:n
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
    for dev in dp.devices
        device!(dev)
        synchronize()
    end
    
    k = dp.masteridx
    for i = 1:length(dp.params)
        if i ≠ k
            device!(k)
            for (master, worker) in zip(dp.params[k], dp.params[i])
                tmpvar = Zeros(typeof(master.delta), master.shape)
                master.delta .+= copyto!(tmpvar, worker.delta)
            end
            device!(dp.devices[i])
            zerograds!(dp.params[i])
        end
    end
    return sum(l)
end

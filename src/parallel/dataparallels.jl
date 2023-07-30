export DataParallelS
export fwdbwd

"""
mutable struct DataParallelS{T} <: Parallel

# Constructor
    DataParallelS{T}(model    :: T;
                     master   :: Int=0,            # master device
                     devices  :: Vector{Int}=[0],  # workers devices
                     criteria :: Function,         # loss function
                     xspliter :: Spliter,          # split input flags for devices
                     yspliter :: Spliter,          # split label flags for devices
                     type     :: Type=CuArray{Float32}) where T
"""
mutable struct DataParallelS{T} <: Parallel
    master   :: Int                      # master device
    imaster  :: Int                      # master device's index
    copylist :: Vector{Int}              # non-master devices index
    calcdevs :: Vector{Int}              # devices that get involed into calculation
    devices  :: Vector{Int}              # all devices
    criteria :: Function                 # loss function
    xspliter :: Spliter                  # split input flags
    yspliter :: Spliter                  # split label flags
    models   :: Vector{T}                # worker-models and master model
    params   :: Vector{Vector{Variable}} # worker-models and master model's params
    caches   :: Vector                   # caches on master device
    tuples   :: Vector{Tuple{Int,Int}}   # params' coords on non-master device
    type     :: Type
    function DataParallelS(model    :: T;
                           master   :: Int=0,
                           workers  :: Vector{Int}=[0],
                           criteria :: Function,
                           xspliter :: Spliter=(dim=1, keptsame=false),
                           yspliter :: Spliter=(dim=1, keptsame=false),
                           type     :: Type=CuArray{Float32}) where T
        imaster  = - 1
        nworkers = length(workers)
        for (i, worker) in enumerate(workers)
            @assert worker ≥ 0 "device ≥ 0 shall be met, but got $worker"
            if worker == master
                imaster = i
                break
            end
        end
        if imaster == - 1
            # master not in workers
            copylist = collect(1:nworkers)
            calcdevs = copy(workers)
            devices  = push!(workers, master)
            imaster  = nworkers + 1
        else
            # master is in workers
            copylist = [i for i in 1:nworkers if i≠imaster]
            calcdevs = copy(workers)
            devices  = workers
        end

        caches = Vector()
        totdev = length(devices)
        tuples = Vector{Dims{2}}(undef,0)
        models = Vector{T}(undef, totdev)
        params = Vector{Vector{Variable{type}}}(undef, totdev)
        @sync for i = 1:totdev
            # switch context and make replications of model and param
            device!(devices[i]) do
                models[i] = clone(model, type=type)
                params[i] = paramsof(models[i])
                if !isequal(i, imaster)
                    # mark worker coords in params
                    for j = 1:length(params[i])
                        push!(tuples, (i,j))
                    end
                end
            end
        end

        device!(devices[imaster]) do
            # caches to hold gradients
            for x in params[imaster]
                push!(caches, zero(ᵛ(x)))
            end
        end

        new{T}(master,
               imaster,
               copylist,
               calcdevs,
               devices,
               criteria,
               xspliter,
               yspliter,
               models,
               params,
               caches,
               tuples,
               type)
    end
end



function Base.show(io::IO, Q::DataParallelS{T}) where T
    workers = ""
    for dev in Q.calcdevs
        workers *= string(dev)
        workers *= " "
    end
    print(io,   "────────────────────────────────────────")
    println("\n DataParallelS{$T}")
    println(io, "────────────────────────────────────────")
    println(io, " master    = $(Q.master)")
    println(io, " workers   = $(workers)")
    println(io, " criteria  = $(Q.criteria)")
    println(io, "  xspliter = $(Q.xspliter)")
    println(io, "  yspliter = $(Q.yspliter)")
    println(io, "     dtype = $(Q.type)")
    println(io, "────────────────────────────────────────")
end


function masterof(Q::DataParallelS)
    device!(Q.master)
    return Q.models[Q.imaster]
end

function Mira.xparamsof(Q::DataParallelS)
    device!(Q.master)
    return xparamsof(Q.models[Q.imaster])
end

#  fwd + loss + bwd + zerograd
function fwdbwd(Q::DataParallelS, x, y)
    T = Q.type
    W = length(Q.calcdevs)  # number of GPUs to work
    l = Vector(undef,    W)  # to store losses
    xdim, xkeptsame = Q.xspliter
    ydim, ykeptsame = Q.yspliter
    I₁, I₂, N = keptdims(x; dim=xdim)
    J₁, J₂, L = keptdims(y; dim=ydim)
    @assert N == L "#input=$N and #label=$L do NOT have the same number of samples"
    batchsize = div(N, W)

    # forward, loss and backward
    @sync for (i, d) in enumerate(Q.calcdevs)
        @async begin
            device!(d) do
                k = getidx(N, batchsize, i)
                input = xkeptsame ? x[I₁,k,I₂] : Variable(x[I₁,k,I₂], type=T)
                label = ykeptsame ? y[J₁,k,J₂] : Variable(y[J₁,k,J₂], type=T)
                v = forward(Q.models[i], input)
                c = Q.criteria(v,        label)
                l[i] = cost(c)
                backward(c)
            end
        end
    end

    P = Q.params       # params on all devices
    M = Q.imaster      # master device's index
    C = length(P[M])    # number of params
    master = Q.master
    caches = Q.caches

    # reduce grad to master-device
    for i in Q.copylist
        @sync for j = 1:C
            @async begin
                # copy non-master-device-grad to caches, then
                # reduce cached grad to master-device's grad
                copyto!(caches[j], δ(P[i][j]))
                device!(master) do
                    P[M][j] ← caches[j]
                end
            end
        end
    end

    # zero grad of worker-devices
    @sync for i in Q.copylist
        device!(Q.devices[i]) do
            for j = 1:C
                δ(P[i][j]) .= 0f0
            end
        end
    end

    device!(master)
    return sum(l) / W
end


function syncparams(Q::DataParallelS)
    P = Q.params
    M = Q.imaster

    # from master-GPU to non-master-GPUs
    @sync for (i, j) in Q.tuples
        @async begin
            copyto!(ᵛ(P[i][j]), ᵛ(P[M][j]))
        end
    end
    
    device!(Q.master)
    return nothing
end

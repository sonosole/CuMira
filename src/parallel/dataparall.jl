abstract type Parallel end


"""
mutable struct DataParallel{T} <: Parallel

# Constructor
    DataParallel{T}(model     :: T;
                    master    :: Int=0,            # master device
                    devices   :: Vector{Int}=[0],  # workers devices
                    criterion :: Function,         # loss function
                    xspliter  :: Function,         # function to split input for devices
                    yspliter  :: Function,         # function to split label for devices
                    type      :: Type=Array{Float32}) where T
"""
mutable struct DataParallel{T} <: Parallel
    masteridx :: Int                      # master device's index
    devices   :: Vector{Int}              # workers devices
    criterion :: Function                 # loss function
    xspliter  :: Function                 # function to split input for devices
    yspliter  :: Function                 # function to split label for devices
    models    :: Vector{T}                # models on `devices`
    mparam    :: Vector{Variable}         # master-device's params
    params    :: Vector{Vector{Variable}} # workers-device's params
    function DataParallel{T}(model     :: T,
                             master    :: Int=0,
                             devices   :: Vector{Int}=[0],
                             criterion :: Function,
                             xspliter  :: Function,
                             yspliter  :: Function,
                             type      :: Type=Array{Float32}) where T
        @assert master in devices "master not in devices"
        ntasks = length(devices)
        models = Vector{T}(undef, ntasks)
        params = Vector{Vector{Variable}}()
        for i  = 1:ntasks
            device!(devices[i])
            models[i] = clone(model, type=type)
            if devices[i] ≠ master
                push!(params, paramsof(models[i]))
            else
                masteridx = i
            end
        end

        new{T}(masteridx,
               devices,
               criterion,
               xspliter,
               yspliter,
               models,
               paramsof(models[masteridx]),
               params)
    end
end


masterof(dp::DataParallel) = dp.models[dp.masteridx]
xparamsof(dp::DataParallel) = xparamsof(masterof(dp))


function fwdbwd(dp::DataParallel, x, y)
    n = length(dp.devices)
    l = Vector(undef, n)
    @sync begin
        Threads.@threads for i = 1:n
            device!(dp.devices[i])
            v = forward(dp.models[i], xspliter(x, i, n))
            c = dp.criterion(v,       yspliter(y, i, n))
            backward(c)
            l[i] = cost(c)
        end
    end

    for i = 1:length(dp.params)
        for (master, worker) in zip(dp.mparam, dp.params[i])
            master.delta .+= worker.delta
        end
        zerograds!(dp.params[i])
    end
    return sum(l)
end

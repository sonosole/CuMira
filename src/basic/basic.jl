include("./CuArrayFun.jl")


export CuBlocks
export CuThreads
export SetBlockSize


# 每个 Block 块里拥有的线程数
global XThreadsPerXBlock = 512
global YThreadsPerYBlock = 512
global ZThreadsPerZBlock = 32


# 设置每个 block 的线程数量
function SetBlockSize(x::Int)
    @assert x≥1 "minimum number of threads is 1, but got $x"
    global XThreadsPerBlock = x
    return nothing
end

function SetBlockSize(x::Int, y::Int)
    @assert x≥1 "minimum number of threads is 1, but got $x"
    @assert y≥1 "minimum number of threads is 1, but got $y"
    global XThreadsPerBlock = x
    global YThreadsPerBlock = y
    return nothing
end

function SetBlockSize(x::Int, y::Int, z::Int)
    @assert x≥1 "minimum number of threads is 1, but got $x"
    @assert y≥1 "minimum number of threads is 1, but got $y"
    @assert z≥1 "minimum number of threads is 1, but got $z"
    global XThreadsPerBlock = x
    global YThreadsPerBlock = y
    global ZThreadsPerBlock = z
    return nothing
end


# 线程数，取决于最大 blocksize 与任务数
function CuThreads(x::Int)
    @assert x≥1 "minimum number of tasks is 1, but got $x"
    return min(x, XThreadsPerBlock)
end

function CuThreads(x::Int, y::Int)
    @assert x≥1 "minimum number of tasks is 1, but got $x"
    @assert y≥1 "minimum number of tasks is 1, but got $y"
    return min(x, XThreadsPerBlock),
           min(y, YThreadsPerBlock)
end

function CuThreads(x::Int, y::Int, z::Int)
    @assert x≥1 "minimum number of tasks is 1, but got $x"
    @assert y≥1 "minimum number of tasks is 1, but got $y"
    @assert z≥1 "minimum number of tasks is 1, but got $z"
    return min(x, XThreadsPerBlock),
           min(y, YThreadsPerBlock),
           min(z, ZThreadsPerBlock)
end


# 块数，取决于总任务数与块大小
function CuBlocks(x::Int)
    @assert x≥1 "minimum number of tasks is 1, but got $x"
    return div(x + XThreadsPerBlock - 1, XThreadsPerBlock)
end

function CuBlocks(x::Int, y::Int)
    @assert x≥1 "minimum number of tasks is 1, but got $x"
    @assert y≥1 "minimum number of tasks is 1, but got $y"
    return div(x + XThreadsPerBlock - 1, XThreadsPerBlock),
           div(y + YThreadsPerBlock - 1, YThreadsPerBlock)
end

function CuBlocks(x::Int, y::Int, z::Int)
    @assert x≥1 "minimum number of tasks is 1, but got $x"
    @assert y≥1 "minimum number of tasks is 1, but got $y"
    @assert z≥1 "minimum number of tasks is 1, but got $z"
    return div(x + XThreadsPerBlock - 1, XThreadsPerBlock),
           div(y + YThreadsPerBlock - 1, YThreadsPerBlock),
           div(z + ZThreadsPerBlock - 1, ZThreadsPerBlock)
end


const culog = CUDA.log
const cuexp = CUDA.exp

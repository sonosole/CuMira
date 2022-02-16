include("./CuArrayFun.jl")


export CuBlocks
export CuThreads
export SetThreadsPerBlock

# 每个 Block 块里拥有的线程数
global ThreadsPerBlock = 512

# 块数，取决于总任务数与块大小
function CuBlocks(n::Int)
    @assert n>=1 "minimum number of tasks is 1, but got $n"
    return div(n + ThreadsPerBlock - 1, ThreadsPerBlock)
end

# 线程数，取决于最大 blocksize 与任务数
function CuThreads(n::Int)
    @assert n>=1 "minimum number of tasks is 1, but got $n"
    return min(n, ThreadsPerBlock)
end

# 设置每个 block 的线程数量
function SetThreadsPerBlock(n::Int)
    @assert n>=1 "minimum number of threads is 1, but got $n"
    global ThreadsPerBlock = n
    return nothing
end


const culog = CUDA.log
const cuexp = CUDA.exp

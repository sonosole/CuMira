export cudafree

function cudafree(;threshold::Int=512, show::Bool=false, full::Bool=false)
    # 1 MiB = 1048576 B
    if CUDA.available_memory() ÷ 1048576 < threshold
        CUDA.reclaim()
        GC.gc(full)
    end
    if show
        CUDA.memory_status()
        println()
    end
end

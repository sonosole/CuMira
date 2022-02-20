"""
    getidx(N::Int, batchsize::Int, k::Int) -> start:final

`N` is the total number of samples, `batchsize` is the number of samples a device
would process, `k` means get the k-th device's samples.
# Example
    julia> getidx(17, 8, 1)
    1:8

    julia> getidx(17, 8, 2)
    9:16

    julia> getidx(17, 8, 3)
    17:17

"""
@inline function getidx(N::Int, batchsize::Int, k::Int)
    start = 1 + (k-1)*batchsize
    final = min(k*batchsize, N)
    return start:final
end


"""
    keptdims(x; dim=1) -> CartesianIndices1, CartesianIndices2, lenofthisdim
"""
@inline function keptdims(x; dim=1)
    sizex = size(x)
    dims  = ndims(x)
    I1 = CartesianIndices(sizex[1:dim-1])
    I2 = CartesianIndices(sizex[dim+1:dims])
    return I1, I2, sizex[dim]
end

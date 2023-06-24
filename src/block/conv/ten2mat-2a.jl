export create_code_ten2mat2a

function create_code_ten2mat2a(D::Int)
    n = "n"
    r = "r"
    k = "k"

    body = """
    function _tensor2matrix2a(y,
                            x,
                            ekernel   :: Dims{$D},
                            dilation  :: Dims{$D},
                            stride    :: Dims{$D},
                            zsize     :: Dims{$D},
                            rows      :: Int,
                            cols      :: Int,
                            xchannels :: Int)

        xthread = blockDim().x * (blockIdx().x - 1) + threadIdx().x
        spacing = blockDim().x * gridDim().x

        for $n = xthread : spacing : cols
            # coords of one patch
            $(ΔcoordsWithb(D, "zsize", n, r))
            @inbounds begin
                j = 0                  # iter index for n-th patch
                o = (n - 1) * rows     # offset for n-th patch
                $(patchloop(D, r, k))
                    for c = 1 : xchannels
                        j = j + 1
                        y[o + j] = $(xelement(D, k))
                    end
                end
            end
        end
        return nothing
    end
    """
    return eval(Meta.parse(body))
end


for D in (1,2,3,4,5)
    create_code_ten2mat2a(D)
end



# x → [im2col] → y → [conv] → z
function ten2mat_nd_infos2a(x        :: CuArray,
                          padding  :: NTuple{D,Dims{2}},
                          kernel   :: Dims{D},
                          dilation :: Dims{D},
                          stride   :: Dims{D}) where D
    assertdim(x, 2+D)
    sizeofx = size(x)
    xsize   = ntuple(i -> sizeofx[i+1] + sum(padding[i]), D)             # equivalent spatial width after padding
    ekernel = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D)          # equivalent kernel size when dialating
    zsize   = ntuple(i -> (xsize[i] - ekernel[i]) ÷ stride[i] + 1, D)    # output spatial width after conv

    xchannels = sizeofx[1]
    batchsize = sizeofx[2+D]
    npatches  = prod(zsize)           # total moving steps along all D dims
    rows = xchannels * prod(kernel)  # total number of elements of a patch, i.e. rows of im2col
    cols = batchsize * npatches      # total moving steps in a batch, i.e. cols of im2col
    return rows, cols, xchannels, ekernel, zsize
end



function tensor2matrix2a(x        :: CuArray{T},
                       padding  :: NTuple{D,NTuple{2,Int}},
                       kernel   :: NTuple{D,Int},
                       dilation :: NTuple{D,Int},
                       stride   :: NTuple{D,Int},
                       padval   :: Real = 0) where {T,D}

    rows, cols, xchannels, ekernel, zsize =
    ten2mat_nd_infos2a(x, padding, kernel, dilation, stride)

    x = padconst(x, ntuple(i -> (1 < i < D+2) ? padding[i-1] : (0,0), D+2), padval)
    y = similar(x, rows, cols)

    @cuda blocks=CuBlocks(cols) threads=CuThreads(cols) (
        _tensor2matrix2a(y, x, ekernel, dilation, stride, zsize, rows, cols, xchannels)
    )

    return y
end

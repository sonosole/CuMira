export create_code_ten2mat1a

function create_code_ten2mat1a(D::Int)
    n = "n"
    r = "r"
    k = "k"
    body = """
    function ten2matfwd1a(y,
                          x,
                          ekernel   :: Dims{$D},
                          dilation  :: Dims{$D},
                          stride    :: Dims{$D},
                          zsize     :: Dims{$D},
                          rows      :: Int,
                          npatches  :: Int,
                          batchsize :: Int,
                          xchannels :: Int)

        elcount = rows * npatches
        xthread = blockDim().x * (blockIdx().x - 1) + threadIdx().x
        spacing = blockDim().x * gridDim().x

        # a thread processes a batched patch
        for $n = xthread : spacing : npatches
            # coords of one patch
            $(Δcoords(D, "zsize", n, r))
            nₒ = (n - 1) * rows  # <---------- offset between columns
            @inbounds for b = 1 : batchsize
                bₒ = (b - 1) * elcount  # <--- offset between samples
                i  = 0
                $(patchloop(D, r, k))
                    for c = 1 : xchannels
                        i = i + 1
                        y[i + nₒ + bₒ] = $(xelement(D, k))
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
    create_code_ten2mat1a(D)
end


# x → [im2col] → y → [conv] → z
function ten2matFwdInfo1a(x        :: CuArray,
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
    npatches  = prod(zsize)          # total moving steps along all D dims
    rows = xchannels * prod(kernel)  # total number of elements of a patch, i.e. rows of im2col
    cols = batchsize * npatches      # total moving steps in a batch, i.e. cols of im2col
    return rows, cols, npatches, batchsize, xchannels, ekernel, zsize
end


# x → [im2col] → y → [conv] → z
function ten2mat1a(x        :: CuArray{T},
                   padding  :: Pads{D},
                   kernel   :: Dims{D},
                   dilation :: Dims{D},
                   stride   :: Dims{D},
                   padmode  :: Function = padconst,
                   padval   :: Real = 0) where {T,D}

    rows, cols, npatches, batchsize, xchannels, ekernel, zsize =
    ten2matFwdInfo1a(x, padding, kernel, dilation, stride)

    if padmode == padconst
        x = padmode(x, extendpad(padding), padval)
    else
        x = padmode(x, extendpad(padding))
    end

    y = similar(x, rows, cols)

    @cuda blocks=CuBlocks(npatches) threads=CuThreads(npatches) (
        ten2matfwd1a(y, x, ekernel, dilation, stride, zsize, rows, npatches, batchsize, xchannels)
    )

    return y
end

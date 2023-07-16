export create_code_ten2mat_fwd
export create_code_ten2mat_bwd


function create_code_ten2mat_fwd(D::Int)
    e    = "e"  # element index str
    n    = "n"  # patch index str
    l    = "l"  # local coords str
    g    = "g"  # global coords str
    k    = "k"  # input's spatial-dims index str
    body = """
    function ten2matfwd(y,
                        x,
                        kernel    :: Dims{$D},   # conv kernel size
                        dilation  :: Dims{$D},   # conv kernel dilation
                        stride    :: Dims{$D},   # conv kernel moving stride
                        zsize     :: Dims{$D},   # feature's spatial size after conv
                        rows      :: Int,
                        leny      :: Int,
                        npatches  :: Int,
                        xchannels :: Int)

        ithread = blockDim().x * (blockIdx().x - 1) + threadIdx().x
        spacing = blockDim().x * gridDim().x

        for m = ithread : spacing : leny
            row = mod(m-1, rows) + 1        # row index of y
            col = div(m-1, rows) + 1        # col index of y
            c = mod(row-1, xchannels) + 1   # channel index
            $e = div(row-1, xchannels) + 1   # element index in a patch
            $n = mod(col-1, npatches) + 1    # patch index
            b = div(col-1, npatches) + 1    # sample index

            @inbounds begin
                # local coords diff inside patch
            $(Δcoords(D, "kernel", e, l, indent="    "))
                # global coords diff between adjacent patchs
            $(Δcoords(D, "zsize",  n, g, indent="    "))
                # absolute coords of x
            $(coords(D, l, g, k))
                y[m] = $(xelement(D, k))
            end
        end
        return nothing
    end
    """
    return eval(Meta.parse(body))
end


function create_code_ten2mat_bwd(D::Int)
    e    = "e"  # element index str
    n    = "n"  # patch index str
    l    = "l"  # local coords str
    g    = "g"  # global coords str
    k    = "k"  # input's spatial-dims index str
    body = """
    function ten2matbwd(y,
                        x,
                        kernel    :: Dims{$D},   # conv kernel size
                        dilation  :: Dims{$D},   # conv kernel dilation
                        stride    :: Dims{$D},   # conv kernel moving stride
                        zsize     :: Dims{$D},   # feature's spatial size after conv
                        rows      :: Int,
                        leny      :: Int,
                        npatches  :: Int,
                        xchannels :: Int)

        ithread = blockDim().x * (blockIdx().x - 1) + threadIdx().x
        spacing = blockDim().x * gridDim().x

        for m = ithread : spacing : leny
            row = mod(m-1, rows) + 1        # row index of y
            col = div(m-1, rows) + 1        # col index of y
            c = mod(row-1, xchannels) + 1   # channel index
            $e = div(row-1, xchannels) + 1   # element index in a patch
            $n = mod(col-1, npatches) + 1    # patch index
            b = div(col-1, npatches) + 1    # sample index

            @inbounds begin
                # local coords diff inside patch
            $(Δcoords(D, "kernel", e, l, indent="    "))
                # global coords diff between adjacent patchs
            $(Δcoords(D, "zsize",  n, g, indent="    "))
                # absolute coords of x
            $(coords(D, l, g, k))
                CUDA.@atomic $(xelement(D, k)) += y[m]
            end
        end
        return nothing
    end
    """
    return eval(Meta.parse(body))
end


for D in (1,2,3,4,5)
    create_code_ten2mat_fwd(D)
    create_code_ten2mat_bwd(D)
end


import Mira.ten2matFwdInfo
function ten2matFwdInfo(x        :: CuArray,
                        padding  :: Pads{D},
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
    ylen = rows * cols
    return rows, cols, ylen, npatches, xchannels, zsize
end


import Mira.ten2mat
function ten2mat(x        :: CuArray{T},
                 padding  :: Pads{D},
                 kernel   :: Dims{D},
                 dilation :: Dims{D},
                 stride   :: Dims{D},
                 padmode  :: Function = padconst,
                 padval   :: Real = 0) where {T,D}

    rows, cols, ylen, npatches, xchannels, zsize =
    ten2matFwdInfo(x, padding, kernel, dilation, stride)

    if padmode == padconst
        x = padmode(x, extendpad(padding), padval)
    else
        x = padmode(x, extendpad(padding))
    end

    y = similar(x, rows, cols)

    @cuda blocks=CuBlocks(ylen) threads=CuThreads(ylen) (
        ten2matfwd(y, x, kernel, dilation, stride, zsize, rows, ylen, npatches, xchannels)
    )

    return y
end


function ten2mat(x        :: Variable{CuArray{T}},
                 padding  :: Pads{D},
                 kernel   :: Dims{D},
                 dilation :: Dims{D},
                 stride   :: Dims{D},
                 padmode  :: Function = padconst,
                 padval   :: Real = 0) where {T,D}

    rows, cols, ylen, npatches, xchannels, zsize =
    ten2matFwdInfo(ᵛ(x), padding, kernel, dilation, stride)

    if padmode == padconst
        x = padmode(x, extendpad(padding), padval)
    else
        x = padmode(x, extendpad(padding))
    end

    y = similar(x, rows, cols)

    @cuda blocks=CuBlocks(ylen) threads=CuThreads(ylen) (
        ten2matfwd(ᵛ(y), ᵛ(x), kernel, dilation, stride, zsize, rows, ylen, npatches, xchannels)
    )

    if y.backprop
        y.backward = function ∇ten2mat()
            if need2computeδ!(x)
                zerodelta(x)
                @cuda blocks=CuBlocks(ylen) threads=CuThreads(ylen) (
                    ten2matbwd(ᵟ(y), ᵟ(x), kernel, dilation, stride, zsize, rows, ylen, npatches, xchannels)
                )
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

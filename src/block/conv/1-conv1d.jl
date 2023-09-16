# in2col forward cuda kernel
function in2col_FCudaKernel(o,
                            x,
                            ichannels::Int,
                            stride::Int,
                            step::Int,        # timeSteps after PlainConv1d operation
                            rows::Int,        # == ichannels * kernel
                            onum::Int)        # == length(o)
    finalIdx = onum
    startIdx = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    strideID = blockDim().x * gridDim().x

    for n = startIdx:strideID:finalIdx # n-th index in o, n ∈ 1:length(o)
        I = mod(n-1,rows)+1            # row index in o
        J = div(n-1,rows)+1            # col index in o

        p = mod(J-1,step)+1            # patch index of b-th batch of x
        b = div(J-1,step)+1            # batch index in x

        i = mod(I-1,ichannels)+1       # row index of b-th batch of x
        t = div(I-1,ichannels)+1       # col index of b-th batch of x in one patch
        t = t + (p - 1)*stride         # col index of s-th patch of b-th batch of x
        @inbounds o[I,J] = x[i,t,b]    # copy data from in to col
    end
    return nothing
end


# in2col backward cuda kernel normal version
# use this when there is overlap between patchs
# that's to say stride < kernel
function in2col_BCudaKernel_Normal(o,
                                   x,
                                   ichannels::Int,
                                   stride::Int,
                                   step::Int,
                                   rows::Int,
                                   nums::Int)
    finalIdx = nums  # == (rows * batchsize)
    strideID = blockDim().x * gridDim().x
    startIdx = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    for τ = 1:step                         # column offset or τ-th time slice
        for n = startIdx:strideID:finalIdx # n-th index in reshape(o,rows,step,batchsize)[:,1,:], n ∈ 1:(rows*batchsize)
            I = mod(n-1,rows)+1            # row index of τ-th step of o in all batchs
            J = div(n-1,rows)*step + τ     # col index of τ-th step of o in all batchs

            p = mod(J-1,step)+1            # patch index of b-th batch of x
            b = div(J-1,step)+1            # batch index in x

            i = mod(I-1,ichannels)+1       # row index of b-th batch of x
            t = div(I-1,ichannels)+1       # col index of b-th batch of x in one patch
            t = t + (p - 1)*stride         # col index of s-th patch of b-th batch of x
            @inbounds x[i,t,b] += o[I,J]   # pass gradient from col to in
        end
        sync_threads()
    end
    return nothing
end


# in2col backward cuda kernel fast version
# use this when there is no overlap between patchs
# that's to say stride>=kernel
function in2col_BCudaKernel_Fast(o,
                                 x,
                                 ichannels::Int,
                                 stride::Int,
                                 step::Int,        # timeSteps after PlainConv1d operation
                                 rows::Int,        # == ichannels * kernel
                                 onum::Int)        # == length(o)
    finalIdx = onum
    strideID = blockDim().x * gridDim().x
    startIdx = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    for n = startIdx:strideID:finalIdx # n-th element in o
        I = mod(n-1,rows)+1            # row index in o
        J = div(n-1,rows)+1            # col index in o

        p = mod(J-1,step)+1            # patch index of b-th batch of x
        b = div(J-1,step)+1            # batch index in x

        i = mod(I-1,ichannels)+1       # row index of b-th batch of x
        t = div(I-1,ichannels)+1       # col index of b-th batch of x in one patch
        t = t + (p - 1)*stride         # col index of s-th patch of b-th batch of x
        @inbounds x[i,t,b] += o[I,J]   # copy data from to col to in
    end
    return nothing
end


# in2col for predict
function Mira.in2col(x::CuArray{T}, kernel::Int, stride::Int) where T
    # from (ichannels,width,batchsize) to (ichannels*kernel,cols)
    (ichannels,width,batchsize) = size(x)
    step = floor(Int,(width-kernel)/stride + 1)
    cols = step * batchsize
    rows = ichannels * kernel
    onum = rows * cols

    y = CUDA.zeros(T, rows, cols)
    @cuda blocks=CuBlocks(onum) threads=CuThreads(onum) (
    in2col_FCudaKernel(y, x, ichannels, stride, step, rows, onum) )
    return y
end


# in2col for training
function Mira.in2col(x::Variable{CuArray{T}}, kernel::Int, stride::Int) where T
    # x from (ichannels,width,batchsize) to (ichannels*kernel,cols)
    (ichannels, width, batchsize) = size(x)
    step = floor(Int,(width-kernel)/stride + 1)
    cols = step * batchsize
    rows = ichannels * kernel
    onum = rows * cols

    y = Variable{CuArray{T}}(CUDA.zeros(T, rows, cols), x.backprop)
    @cuda blocks=CuBlocks(onum) threads=CuThreads(onum) (
    in2col_FCudaKernel(ᵛ(y), ᵛ(x), ichannels, stride, step, rows, onum) )

    if y.backprop
        y.backward = function in2colBackward()
            if needgrad(x)
                if stride < kernel
                    nums = rows * batchsize
                    @cuda blocks=CuBlocks(nums) threads=CuThreads(nums) (
                    in2col_BCudaKernel_Normal(δ(y), δ(x), ichannels, stride, step, rows, nums) )
                else
                    @cuda blocks=CuBlocks(onum) threads=CuThreads(onum) (
                    in2col_BCudaKernel_Fast(δ(y), δ(x), ichannels, stride, step, rows, onum) )
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

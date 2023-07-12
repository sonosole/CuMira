import Mira.assertdim
import Mira.ten2matFwdInfo
import Mira.ten2mat


function patchloop(D::Int, δ::String, k::String)
    spans = ntuple(D) do d
        i = lower_corner_digits(d)
        O = "$δ$i*stride[$d]"     # offset for d-th dim
        if d == 1
            return "$O + 1 : dilation[$d] : ekernel[$d] + $O"
        else
            return "$O + 1 : dilation[$d] : ekernel[$d] + $O,"
        end
    end

    i = lower_corner_digits(D)
    loop = """
    for $k$i = $(spans[D])
    """
    for d in D-1:-1:1
        i = lower_corner_digits(d)
        loop *= """
                        $k$i = $(spans[d])
        """
    end

    return loop
end


function xelement(D::Int, k::String)
    x = "x[c, "
    for d in 1:D
        i  = lower_corner_digits(d)
        x *= "$k$i, "
    end
    x *= "b]"
    return x
end


"""
    Δcoords(D::Int, shape::String, n::String, δ::String) -> codestr::String

The returned code computes the difference between coordinates of the `n`-th
element and the `1`-st element in a `D`-dimentional array of size `shape`.
For example, an array of size (2, 3),\n
    1  3  5
    2  4  6
the `6`-th element's cartesian coords is (2,3), and the `1`-st element's
coords is (1,1), so the coordinates difference is (1,2) ← (2,3) .- (1,1)

+ `shape` is the spatial dimentions of an array
+ `D` is the number of spatial dimentions of this array
+ `n` is the `n`-th element in this array
+ `δ` is the prefix of the names for the difference between coordinates

# Example
julia> Δcoords(3, "zsize", "n", "delta") |> println
t      = n - 1
        delta₁ = mod(t, zsize[1])    # rows-1 at dims 1
        t      = div(t, zsize[1])    # cols-1 at dims 2
        delta₂ = mod(t, zsize[2])    # rows-1 at dims 2
        t      = div(t, zsize[2])    # cols-1 at dims 3
        delta₃ = mod(t, zsize[3])    # rows-1 at dims 3
"""
function Δcoords(D::Int, shape::String, n::String, δ::String; indent=nothing)
    if isnothing(indent)
        indent = ""
    end

    s = repeat(" ", length(δ))
    coords = """
    $(indent)t$s = $n - 1
    """

    for d in 1:D-1
        i = lower_corner_digits(d)
        s = repeat(" ", length(i) + length(δ) - 1)
        r = "$δ$i"
        t = "t$s"
        coords *= """
                $(indent)$r = mod(t, $shape[$d])    # rows-1 at dims $d
                $(indent)$t = div(t, $shape[$d])    # cols-1 at dims $(d+1)
        """
    end

    i = lower_corner_digits(D)
    r = "$δ$i"
    coords *= """
            $(indent)$r = mod(t, $shape[$D])    # rows-1 at dims $D
    """

    return coords
end



"""
    ΔcoordsWithb(D::Int, shape::String, n::String, δ::String) -> codestr::String

The returned code computes two things:
+ the difference between the first `D`-coordinates of the `n`-th element and the `1`-st element in the `(D+1)`-dimentions of `array`
+ the sample index of the `n`-th element of `array`, i.e. the index in the batch dimention
For example, an array of size `(3, 3, 2)`,\n
    3×3×2 Array{Int64, 3}:
    [:, :, 1] =
     1  4  7
     2  5  8
     3  6  9

    [:, :, 2] =
     10  13  16
     11  14  17
     12  15  18
the `15`-th element's cartesian coords is `(3,2,2)`, and the `1`-st element's coords is `(1,1,1)`, so the first `D` (which is 2)\n
coordinates difference is `(2,1) ← (3,2) .- (1,1)`, and the index of batch dimention for the `15`-th element is `2`.

+ `shape` is the spatial dimentions of an array
+ `D` is the number of spatial dimentions of this array
+ `n` is the `n`-th element in this array
+ `δ` is the prefix of the names for the difference between coordinates

# Example
julia> ΔcoordsWithb(4, "zsize", "m", "δ") |> println
t  = m - 1
        δ₁ = mod(t, zsize[1])      # rows-1 at dims 1
        t  = div(t, zsize[1])      # cols-1 at dims 2
        δ₂ = mod(t, zsize[2])      # rows-1 at dims 2
        t  = div(t, zsize[2])      # cols-1 at dims 3
        δ₃ = mod(t, zsize[3])      # rows-1 at dims 3
        t  = div(t, zsize[3])      # cols-1 at dims 4
        δ₄ = mod(t, zsize[4])      # rows-1 at dims 4
        b  = div(t, zsize[4]) + 1  # batchsize
"""
function ΔcoordsWithb(D::Int, shape::String, n::String, δ::String)
    s = repeat(" ", length(δ))

    coords = """
    t$s = $n - 1
    """

    for d in 1:D-1
        i = lower_corner_digits(d)
        s = repeat(" ", length(i) + length(δ) - 1)
        r = "$δ$i"
        t = "t$s"
        coords *= """
                $r = mod(t, $shape[$d])      # rows-1 at dims $d
                $t = div(t, $shape[$d])      # cols-1 at dims $(d+1)
        """
    end

    i = lower_corner_digits(D)
    s = repeat(" ", length(i) + length(δ) - 1)
    r = "$δ$i"
    b = "b$s"
    coords *= """
            $r = mod(t, $shape[$D])      # rows-1 at dims $D
            $b = div(t, $shape[$D]) + 1  # batchsize
    """

    return coords
end


"""
    coords(D::Int, localδx::String, globalδx::String, k::String)
Absolute coords inside a patch.

+ `D` is ndims of the patch
+ `localδx` is the local difference of coords between one element and the 1-st element
+ `globalδx` is the difference of coords between one patch and the 1-st patch
+ `k` is the prefix of absolute coords names
"""
function coords(D::Int, localδx::String, globalδx::String, k::String)
    i = lower_corner_digits(1)
    body = """
        $k$i = 1 + dilation[1]*$localδx$i + stride[1]*$globalδx$i
    """

    for d in 2:D
        i = lower_corner_digits(d)
        body *= """
                    $k$i = 1 + dilation[$d]*$localδx$i + stride[$d]*$globalδx$i
        """
    end
    return body
end

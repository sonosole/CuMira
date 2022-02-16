export CuLogSum2Exp
export CuLogSum3Exp
export CuLogSum4Exp
export CuLogSumExp


function CuLogSum2Exp(a, b)
    isinf(a) && return b
    isinf(b) && return a
    if a < b
        a, b = b, a
    end
    return (a + culog(1.0 + cuexp(b-a)))
end


function CuLogSum3Exp(a::Real, b::Real, c::Real)
    return CuLogSum2Exp(CuLogSum2Exp(a,b),c)
end


function CuLogSum4Exp(a::Real, b::Real, c::Real, d::Real)
    return CuLogSum2Exp(CuLogSum2Exp(CuLogSum2Exp(a,b),c),d)
end


function CuLogSumExp(a)
    tmp = LogZero(eltype(a))
    for i = 1:length(a)
        tmp = CuLogSum2Exp(tmp, a[i])
    end
    return tmp
end

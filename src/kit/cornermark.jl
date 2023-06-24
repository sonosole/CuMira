"""
    lower_corner_digits(n::Int)
Convert the decimal number to lower right corner digits in the form of string

# Example
    julia> "X" * lower_corner_digits(321)
    "X₃₂₁"
"""
function lower_corner_digits(n::Int)
    mark = Dict(
        0=>"₀",
        1=>"₁",
        2=>"₂",
        3=>"₃",
        4=>"₄",
        5=>"₅",
        6=>"₆",
        7=>"₇",
        8=>"₈",
        9=>"₉"
    )

    s = ""
    j = n
    while div(j, 10) ≠ 0
        g = mod(j, 10)
        s *= mark[g]
        j = div(j, 10)
    end
    g  = mod(j, 10)
    s *= mark[g]
    return reverse(s)
end


"""
    upper_corner_digits(n::Int)
Convert the decimal number to lower right corner digits in the form of string
# Example
    julia> "X" * upper_corner_digits(321)
    "X³²¹"
"""
function upper_corner_digits(n::Int)
    mark = Dict(
        0=>"⁰",
        1=>"¹",
        2=>"²",
        3=>"³",
        4=>"⁴",
        5=>"⁵",
        6=>"⁶",
        7=>"⁷",
        8=>"⁸",
        9=>"⁹"
    )

    s = ""
    j = n
    while div(j, 10) ≠ 0
        g = mod(j, 10)
        s *= mark[g]
        j = div(j, 10)
    end
    g  = mod(j, 10)
    s *= mark[g]
    return reverse(s)
end

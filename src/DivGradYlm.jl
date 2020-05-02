module DivGradYlm

using HCubature
using StaticArrays



export lmax
const lmax = 2



export bitsign
bitsign(b::Bool) = b ? -1 : 1
bitsign(i::Integer) = bitsign(isodd(i))

export bitconj
bitconj(b, x) = x
bitconj(b, x::Complex) = Complex(real(x), bitsign(b) * imag(x))



export sphere_dot
function sphere_dot(::Type{T}, f::F, g::G) where {T, F, G}
    atol = sqrt(eps(T))
    function k(x)
        θ,ϕ = x
        conj(f(θ,ϕ)) * g(θ,ϕ) * sin(θ)
    end
    I,E = hcubature(k, (0,0), (T(π),2*T(π)), atol=atol)
    @assert E <= atol
    I
end

export sphere_vdot
function sphere_vdot(::Type{T}, f::F, g::G) where {T, F, G}
    atol = sqrt(eps(T))
    function k(x)
        θ,ϕ = x
        # We scale the ϕ components, so the metric is trivial
        sum(conj(f(θ,ϕ)) .* g(θ,ϕ)) * sin(θ)
    end
    I,E = hcubature(k, (0,0), (T(π),2*T(π)), atol=atol)
    @assert E <= atol
    I
end

export sphere_tdot
function sphere_tdot(::Type{T}, f::F, g::G) where {T, F, G}
    atol = sqrt(eps(T))
    function k(x)
        θ,ϕ = x
        # We scale the ϕ components, so the metric is trivial
        sum(conj(f(θ,ϕ)) .* g(θ,ϕ)) * sin(θ)
    end
    I,E = hcubature(k, (0,0), (T(π),2*T(π)), atol=atol)
    @assert E <= atol
    I
end



# Metric on the sphere:
#     γ = diag(1, sinθ^2)
# Christoffel symbols on the sphere:
# <https://einsteinrelativelyeasy.com/index.php/general-relativity/34-christoffel-symbol-exercise-calculation-in-polar-coordinates-part-ii>:
#     Γ^θ = [0 0; 0 -sinθ*cosθ]
#     Γ^ϕ = [0 cosθ/sinθ; cosθ/sinθ 0]

# Note: Tensor components have their ϕ component divided by sinθ.

# dY: (∂θ, 1/sinθ ∂ϕ)
# ddY: (∂θ∂θ, 1/sinθ ∂θ∂ϕ, 1/sinθ^2 ∂ϕ∂ϕ)

Y00(θ,ϕ) = sqrt(1/4π)
dY00(θ,ϕ) = (0, 0)
ddY00(θ,ϕ) = (0, 0, 0)

Y10(θ,ϕ) = sqrt(3/4π) * cos(θ)
Y11(θ,ϕ) = -sqrt(3/8π) * sin(θ) * cis(ϕ)
dY10(θ,ϕ) = (-sqrt(3/4π) * sin(θ), 0)
dY11(θ,ϕ) = (-sqrt(3/8π) * cos(θ) * cis(ϕ), -sqrt(3/8π) * im * cis(ϕ))
ddY10(θ,ϕ) = (-sqrt(3/4π) * cos(θ), 0, -sqrt(3/4π) * cos(θ))
ddY11(θ,ϕ) = (sqrt(3/8π) * sin(θ) * cis(ϕ), 0, sqrt(3/8π) * sin(θ) * cis(ϕ))

Y20(θ,ϕ) = sqrt(5/16π) * (-1 + 3*cos(θ)^2)
Y21(θ,ϕ) = -sqrt(15/8π) * cos(θ) * sin(θ) * cis(ϕ)
Y22(θ,ϕ) = sqrt(15/32π) * sin(θ)^2 * cis(2ϕ)
dY20(θ,ϕ) = (-sqrt(45/4π) * cos(θ) * sin(θ), 0)
dY21(θ,ϕ) = (-sqrt(15/8π) * cos(2θ) * cis(ϕ),
             -sqrt(15/8π) * im * cos(θ) * cis(ϕ))
dY22(θ,ϕ) = (sqrt(15/8π) * cos(θ) * sin(θ) * cis(2ϕ),
             sqrt(15/8π) * im * sin(θ) * cis(2ϕ))
ddY20(θ,ϕ) = (-sqrt(45/4π) * cos(2θ), 0, -sqrt(45/4π) * cos(θ)^2)
ddY21(θ,ϕ) = (sqrt(30/π) * cos(θ) * sin(θ) * cis(ϕ),
              sqrt(15/8π) * im * sin(θ) * cis(ϕ),
              sqrt(15/2π) * cos(θ) * sin(θ) * cis(ϕ))
ddY22(θ,ϕ) = (sqrt(15/8π) * cos(2θ) * cis(2ϕ),
              sqrt(15/8π) * im * cos(θ) * cis(2ϕ),
              sqrt(15/32π) * (cos(2θ) - 3) * cis(2ϕ))



export Ylm
function Ylm(l,m,θ::T,ϕ::T)::Complex{T} where {T}
    c = m<0
    s = m<0 ? bitsign(m) : 1
    m = abs(m)
    (l,m) == (0,0) && return s * bitconj(c, Y00(θ,ϕ))
    (l,m) == (1,0) && return s * bitconj(c, Y10(θ,ϕ))
    (l,m) == (1,1) && return s * bitconj(c, Y11(θ,ϕ))
    (l,m) == (2,0) && return s * bitconj(c, Y20(θ,ϕ))
    (l,m) == (2,1) && return s * bitconj(c, Y21(θ,ϕ))
    (l,m) == (2,2) && return s * bitconj(c, Y22(θ,ϕ))
    @error "oops"
end

export gradYlm
# (∂θ, 1/sinθ ∂ϕ)
function gradYlm(l,m,θ::T,ϕ::T)::SVector{2,Complex{T}} where {T}
    c = m<0
    s = m<0 ? bitsign(m) : 1
    m = abs(m)
    (l,m) == (0,0) && return bitconj.(c, s.*dY00(θ,ϕ))
    (l,m) == (1,0) && return bitconj.(c, s.*dY10(θ,ϕ))
    (l,m) == (1,1) && return bitconj.(c, s.*dY11(θ,ϕ))
    (l,m) == (2,0) && return bitconj.(c, s.*dY20(θ,ϕ))
    (l,m) == (2,1) && return bitconj.(c, s.*dY21(θ,ϕ))
    (l,m) == (2,2) && return bitconj.(c, s.*dY22(θ,ϕ))
    @error "oops"
end

export curlYlm
# (-1/sinθ ∂ϕ, ∂θ)
function curlYlm(l,m,θ::T,ϕ::T)::SVector{2,Complex{T}} where {T}
    dr = gradYlm(l,m,θ,ϕ)
    dr[2], -dr[1]
end

export traceYlm
function traceYlm(l,m,θ::T,ϕ::T)::SMatrix{2,2,Complex{T}} where {T}
    (1, 0, 0, 1) .* Ylm(l,m,θ,ϕ)
end

export gradgradYlm
function gradgradYlm(l,m,θ::T,ϕ::T)::SMatrix{2,2,Complex{T}} where {T}
    c = m<0
    s = m<0 ? bitsign(m) : 1
    m = abs(m)
    ddY = nothing
    (l,m) == (0,0) && (ddY = bitconj.(c, s.*ddY00(θ,ϕ)))
    (l,m) == (1,0) && (ddY = bitconj.(c, s.*ddY10(θ,ϕ)))
    (l,m) == (1,1) && (ddY = bitconj.(c, s.*ddY11(θ,ϕ)))
    (l,m) == (2,0) && (ddY = bitconj.(c, s.*ddY20(θ,ϕ)))
    (l,m) == (2,1) && (ddY = bitconj.(c, s.*ddY21(θ,ϕ)))
    (l,m) == (2,2) && (ddY = bitconj.(c, s.*ddY22(θ,ϕ)))
    ddY isa Nothing && @error "oops"
    gg = SMatrix{2,2,Complex{T}}(ddY[1], ddY[2], ddY[2], ddY[3])
    # Note: The Sandberg paper wants this, but this doesn't work
    # gg += l*(l+1)÷2 * traceYlm(l,m,θ,ϕ)
    # Make basis explicitly trace-free
    trgg = gg[1,1] + gg[2,2]
    gg -= trgg/2 * SMatrix{2,2,T}(1, 0, 0, 1)
    # trgg = gg[1,1] + gg[2,2]
    # @assert abs(trgg) <= sqrt(eps())
    gg
end

export epsilonYlm
function epsilonYlm(l,m,θ::T,ϕ::T)::SMatrix{2,2,Complex{T}} where {T}
    (0, 1, -1, 0) .* Ylm(l,m,θ,ϕ)
end

export gradcurlYlm



export AbstractModes
abstract type AbstractModes{T} end
export ScalarModes
struct ScalarModes{T} <: AbstractModes{T}
    m::Dict{NTuple{2,Int}, Complex{T}}
    ScalarModes{T}() where {T} = new{T}(Dict{NTuple{2,Int}, Complex{T}}())
end
export GradModes
struct GradModes{T} <: AbstractModes{T}
    m::Dict{NTuple{2,Int}, Complex{T}}
    GradModes{T}() where {T} = new{T}(Dict{NTuple{2,Int}, Complex{T}}())
end
export CurlModes
struct CurlModes{T} <: AbstractModes{T}
    m::Dict{NTuple{2,Int}, Complex{T}}
    CurlModes{T}() where {T} = new{T}(Dict{NTuple{2,Int}, Complex{T}}())
end
export TraceModes
struct TraceModes{T} <: AbstractModes{T}
    m::Dict{NTuple{2,Int}, Complex{T}}
    TraceModes{T}() where {T} = new{T}(Dict{NTuple{2,Int}, Complex{T}}())
end
export GradGradModes
struct GradGradModes{T} <: AbstractModes{T}
    m::Dict{NTuple{2,Int}, Complex{T}}
    GradGradModes{T}() where {T} = new{T}(Dict{NTuple{2,Int}, Complex{T}}())
end

Base.getindex(m::AbstractModes, x...) = getindex(m.m, x...)
Base.setindex!(m::AbstractModes, x...) = setindex!(m.m, x...)
Base.get(m::AbstractModes, x...) = get(m.m, x...)

lmin(::Type{<:ScalarModes}) = 0
lmin(::Type{<:GradModes}) = 1
lmin(::Type{<:CurlModes}) = 1
lmin(::Type{<:TraceModes}) = 0
lmin(::Type{<:GradGradModes}) = 2

function Base.zero(::Type{M}) where {M<:AbstractModes{T}} where {T}
    modes = M()
    for l in lmin(M):lmax, m in -l:l
        modes[(l,m)] = 0
    end
    modes
end

function Base.rand(::Type{M}) where {M<:AbstractModes{T}} where {T}
    modes = M()
    for l in lmin(M):lmax, m in -l:l
        modes[(l,m)] = rand(T)
    end
    modes
end



export expand_scalar
function expand_scalar(::Type{T}, f::F)::ScalarModes{T} where {T, F}
    atol = sqrt(eps(T))
    modes = ScalarModes{T}()
    for l in 0:lmax, m in -l:l
        modes[(l,m)] = sphere_dot(T, (θ,ϕ) -> Ylm(l,m,θ,ϕ), f)
    end
    modes
end

export eval_scalar
function eval_scalar(modes::ScalarModes{T}, θ,ϕ)::Complex{T} where {T}
    r = Complex{T}(0)
    for l in 0:lmax, m in -l:l
        r += modes[(l,m)] * Ylm(l,m,θ,ϕ)
    end
    r
end

export expand_grad
function expand_grad(::Type{T}, f::F)::GradModes{T} where {T, F}
    atol = sqrt(eps(T))
    modes = GradModes{T}()
    for l in 1:lmax, m in -l:l
        modes[(l,m)] =
            sphere_vdot(T, (θ,ϕ) -> gradYlm(l,m,θ,ϕ), f) / (l*(l+1))
    end
    modes
end

export eval_grad
function eval_grad(modes::GradModes{T}, θ,ϕ)::SVector{2,Complex{T}} where {T}
    r = SVector{2,Complex{T}}(0, 0)
    for l in 1:lmax, m in -l:l
        r += modes[(l,m)] .* gradYlm(l,m,θ,ϕ)
    end
    r
end

export expand_curl
function expand_curl(::Type{T}, f::F)::CurlModes{T} where {T, F}
    atol = sqrt(eps(T))
    modes = CurlModes{T}()
    for l in 1:lmax, m in -l:l
        modes[(l,m)] =
            sphere_vdot(T, (θ,ϕ) -> curlYlm(l,m,θ,ϕ), f) / (l*(l+1))
    end
    modes
end

export eval_curl
function eval_curl(modes::CurlModes{T}, θ,ϕ)::SVector{2,Complex{T}} where {T}
    r = SVector{2,Complex{T}}(0, 0)
    for l in 1:lmax, m in -l:l
        r += modes[(l,m)] .* curlYlm(l,m,θ,ϕ)
    end
    r
end

export expand_trace
function expand_trace(::Type{T}, f::F)::TraceModes{T} where {T, F}
    atol = sqrt(eps(T))
    modes = TraceModes{T}()
    for l in 0:lmax, m in -l:l
        modes[(l,m)] =
            sphere_tdot(T, (θ,ϕ) -> traceYlm(l,m,θ,ϕ), f) / 2
    end
    modes
end

export eval_trace
function eval_trace(modes::TraceModes{T}, θ,ϕ
                    )::SMatrix{2,2,Complex{T}} where {T}
    r = SMatrix{2,2,Complex{T}}(0, 0, 0, 0)
    for l in 0:lmax, m in -l:l
        r += modes[(l,m)] .* traceYlm(l,m,θ,ϕ)
    end
    r
end

export expand_gradgrad
function expand_gradgrad(::Type{T}, f::F)::GradGradModes{T} where {T, F}
    atol = sqrt(eps(T))
    modes = GradGradModes{T}()
    for l in 2:lmax, m in -l:l
        modes[(l,m)] =
            sphere_tdot(T, (θ,ϕ) -> gradgradYlm(l,m,θ,ϕ), f) /
            (l*(l+1) * (l*(l+1)÷2 - 1))
    end
    modes
end

export eval_gradgrad
function eval_gradgrad(modes::GradGradModes{T}, θ,ϕ
                    )::SMatrix{2,2,Complex{T}} where {T}
    r = SMatrix{2,2,Complex{T}}(0, 0, 0, 0)
    for l in 2:lmax, m in -l:l
        r += modes[(l,m)] .* gradgradYlm(l,m,θ,ϕ)
    end
    r
end



export grad_scalar
function grad_scalar(smodes::ScalarModes{T})::GradModes{T} where {T}
    gmodes = GradModes{T}()
    for l in 1:lmax, m in -l:l
        gmodes[(l,m)] = smodes[(l,m)]
    end
    gmodes
end

export div_scalar
function div_scalar(smodes::ScalarModes{T})::ScalarModes{T} where {T}
    dmodes = ScalarModes{T}()
    for l in 0:lmax, m in -l:l
        dmodes[(l,m)] = 2 * smodes[(l,m)]
    end
    dmodes
end

export div_grad
function div_grad(gmodes::GradModes{T})::ScalarModes{T} where {T}
    smodes = ScalarModes{T}()
    smodes[(0,0)] = 0
    for l in 1:lmax, m in -l:l
        smodes[(l,m)] = -(l*(l+1)) * gmodes[(l,m)]
    end
    smodes
end

# div_curl is by definition always zero

export curl_scalar
function curl_scalar(smodes::ScalarModes{T})::CurlModes{T} where {T}
    cmodes = CurlModes{T}()
    for l in 1:lmax, m in -l:l
        cmodes[(l,m)] = - smodes[(l,m)]
    end
    cmodes
end

export curl_grad
function curl_grad(gmodes::GradModes{T})::CurlModes{T} where {T}
    cmodes = CurlModes{T}()
    for l in 1:lmax, m in -l:l
        cmodes[(l,m)] = gmodes[(l,m)]
    end
    cmodes
end

export curl_curl
function curl_curl(cmodes::CurlModes{T}
                   )::Tuple{ScalarModes{T}, GradModes{T}} where {T}
    smodes = ScalarModes{T}()
    smodes[(0,0)] = 0
    for l in 1:lmax, m in -l:l
        smodes[(l,m)] = -l*(l+1) * cmodes[(l,m)]
    end
    gmodes = GradModes{T}()
    for l in 1:lmax, m in -l:l
        gmodes[(l,m)] = - cmodes[(l,m)]
    end
    smodes, gmodes
end

export laplace_scalar
function laplace_scalar(smodes::ScalarModes{T})::ScalarModes{T} where {T}
    div_grad(grad_scalar(smodes))
end

end

module DivGradYlm

using HCubature
using StaticArrays



export lmax
const lmax = 2



export bitsign
bitsign(b::Bool) = b ? -1 : 1
bitsign(i::Integer) = bitsign(isodd(i))



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
        sum(conj(f(θ,ϕ)) .* g(θ,ϕ)) * sin(θ)
    end
    I,E = hcubature(k, (0,0), (T(π),2*T(π)), atol=atol)
    @assert E <= atol
    I
end



Y00(θ,ϕ) = sqrt(1/4π)
gradY00(θ,ϕ) = (0, 0)

Y10(θ,ϕ) = sqrt(3/4π) * cos(θ)
Y11(θ,ϕ) = -sqrt(3/8π) * sin(θ) * cis(ϕ)
gradY10(θ,ϕ) = (-sqrt(3/4π) * sin(θ), 0)
gradY11(θ,ϕ) = (-sqrt(3/8π) * cos(θ) * cis(ϕ),
                -sqrt(3/8π) * im * cis(ϕ))

Y20(θ,ϕ) = sqrt(5/16π) * (-1 + 3*cos(θ)^2)
Y21(θ,ϕ) = -sqrt(15/8π) * cos(θ) * sin(θ) * cis(ϕ)
Y22(θ,ϕ) = sqrt(15/32π) * sin(θ)^2 * cis(2ϕ)
gradY20(θ,ϕ) = (-sqrt(45/4π) * cos(θ) * sin(θ), 0)
gradY21(θ,ϕ) = (-sqrt(15/8π) * cos(2θ) * cis(ϕ),
                -sqrt(15/8π) * im * cos(θ) * cis(ϕ))
gradY22(θ,ϕ) = (sqrt(15/8π) * cos(θ) * sin(θ) * cis(2ϕ),
                sqrt(15/8π) * im * sin(θ) * cis(2ϕ))



export Ylm
function Ylm(l,m,θ::T,ϕ::T)::Complex{T} where {T}
    ϕ′ = bitsign(m<0) * ϕ
    s = m<0 ? bitsign(m) : 1
    m = abs(m)
    (l,m) == (0,0) && return s*Y00(θ,ϕ′)
    (l,m) == (1,0) && return s*Y10(θ,ϕ′)
    (l,m) == (1,1) && return s*Y11(θ,ϕ′)
    (l,m) == (2,0) && return s*Y20(θ,ϕ′)
    (l,m) == (2,1) && return s*Y21(θ,ϕ′)
    (l,m) == (2,2) && return s*Y22(θ,ϕ′)
    @error "oops"
end

export gradYlm
# (∂θ, 1/sinθ ∂ϕ)
function gradYlm(l,m,θ::T,ϕ::T)::SVector{2,Complex{T}} where {T}
    ϕ′ = bitsign(m<0) * ϕ
    s = m<0 ? bitsign(m) : 1
    m = abs(m)
    (l,m) == (0,0) && return s.*gradY00(θ,ϕ′)
    (l,m) == (1,0) && return s.*gradY10(θ,ϕ′)
    (l,m) == (1,1) && return s.*gradY11(θ,ϕ′)
    (l,m) == (2,0) && return s.*gradY20(θ,ϕ′)
    (l,m) == (2,1) && return s.*gradY21(θ,ϕ′)
    (l,m) == (2,2) && return s.*gradY22(θ,ϕ′)
    @error "oops"
end

export curlYlm
# (-1/sinθ ∂ϕ, ∂θ)
function curlYlm(l,m,θ::T,ϕ::T)::SVector{2,Complex{T}} where {T}
    dr = gradYlm(l,m,θ,ϕ)
    (-dr[2], dr[1])
end



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
Base.getindex(m::AbstractModes, x...) = getindex(m.m, x...)
Base.setindex!(m::AbstractModes, x...) = setindex!(m.m, x...)
Base.get(m::AbstractModes, x...) = get(m.m, x...)



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



export grad_scalar
function grad_scalar(smodes::Dict{NTuple{2,Int}, T}) where {T}
    gmodes = Dict{NTuple{2,Int}, Complex{T}}()
    for l in 1:lmax, m in -l:l
        gmodes[(l,m)] = smodes[(l,m)]
    end
    gmodes
end

export curl_scalar
function curl_scalar(smodes::Dict{NTuple{2,Int}, T}) where {T}
    cmodes = Dict{NTuple{2,Int}, Complex{T}}()
    for l in 1:lmax, m in -l:l
        cmodes[(l,m)] = -l*(l+1)*smodes[(l,m)]
    end
    cmodes
end

export div_scalar
function div_scalar(smodes::Dict{NTuple{2,Int}, T}) where {T}
    dmodes = Dict{NTuple{2,Int}, Complex{T}}()
    for l in 1:lmax, m in -l:l
        dmodes[(l,m)] = 2*smodes[(l,m)]
    end
    dmodes
end

export curl_grad
function curl_grad(gmodes::Dict{NTuple{2,Int}, T}) where {T}
    smodes = Dict{NTuple{2,Int}, Complex{T}}()
    for l in 1:lmax, m in -l:l
        smodes[(l,m)] = gmodes[(l,m)]
    end
    smodes
end

end

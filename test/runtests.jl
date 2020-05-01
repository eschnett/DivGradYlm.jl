using DivGradYlm

using Test



@testset "Ylm" begin
    T = Float64
    atol = sqrt(eps(T))
    for l in 0:lmax, m in -l:l, l′=0:lmax, m′=-l′:l′
        I = sphere_dot(Float64,
                       (θ,ϕ) -> Ylm(l,m,θ,ϕ),
                       (θ,ϕ) -> Ylm(l′,m′,θ,ϕ))
        δ = (l,m)==(l′,m′)
        @test abs(I - δ) <= 10*atol
    end
end

@testset "gradYlm" begin
    T = Float64
    atol = sqrt(eps(T))
    for l in 0:lmax, m in -l:l, l′=0:lmax, m′=-l′:l′
        I = sphere_vdot(Float64,
                        (θ,ϕ) -> gradYlm(l,m,θ,ϕ),
                        (θ,ϕ) -> gradYlm(l′,m′,θ,ϕ))
        δ = (l,m)==(l′,m′)
        @test abs(I - l*(l+1)*δ) <= 10*atol
    end
end

@testset "curlYlm" begin
    T = Float64
    atol = sqrt(eps(T))
    for l in 0:lmax, m in -l:l, l′=0:lmax, m′=-l′:l′
        I = sphere_vdot(Float64,
                        (θ,ϕ) -> curlYlm(l,m,θ,ϕ),
                        (θ,ϕ) -> curlYlm(l′,m′,θ,ϕ))
        δ = (l,m)==(l′,m′)
        @test abs(I - l*(l+1)*δ) <= 10*atol
    end
end

@testset "grad/curlYlm" begin
    T = Float64
    atol = sqrt(eps(T))
    for l in 0:lmax, m in -l:l, l′=0:lmax, m′=-l′:l′
        I = sphere_vdot(Float64,
                        (θ,ϕ) -> gradYlm(l,m,θ,ϕ),
                        (θ,ϕ) -> curlYlm(l′,m′,θ,ϕ))
        @test abs(I) <= 10*atol
    end
end

@testset "expand_scalar" begin
    T = Float64
    atol = sqrt(eps(T))
    for l in 0:lmax, m in -l:l
        f(θ,ϕ) = Ylm(l,m,θ,ϕ)
        modes = expand_scalar(T, f)
        modes::ScalarModes{T}
        for l′=0:lmax, m′=-l′:l′
            δ = (l,m)==(l′,m′)
            @test abs(modes[(l′,m′)] - δ) <= 10*atol
        end
    end
end

@testset "expand coordinates" begin
    T = Float64
    atol = sqrt(eps(T))
    x(θ,ϕ) = sin(θ) * cos(ϕ)
    y(θ,ϕ) = sin(θ) * sin(ϕ)
    z(θ,ϕ) = cos(θ)
    mx = Dict((1,-1) => sqrt(2π/3), (1,1) => -sqrt(2π/3))
    my = Dict((1,-1) => sqrt(2π/3)*im, (1,1) => sqrt(2π/3)*im)
    mz = Dict((1,0) => sqrt(4π/3))
    mx′ = expand_scalar(T, x)
    my′ = expand_scalar(T, y)
    mz′ = expand_scalar(T, z)
    for l in 0:lmax, m in -l:l
        @test abs(mx′[(l,m)] - get(mx, (l,m), 0)) <= 10*atol
        @test abs(my′[(l,m)] - get(my, (l,m), 0)) <= 10*atol
        @test abs(mz′[(l,m)] - get(mz, (l,m), 0)) <= 10*atol
    end
end

@testset "eval_scalar" begin
    T = Float64
    atol = sqrt(eps(T))
    # Choose function
    modes = ScalarModes{T}()
    for l in 0:lmax, m in -l:l
        modes[(l,m)] = rand(Complex{T})
    end
    f(θ,ϕ) = eval_scalar(modes, θ,ϕ)
    # Expand into modes
    modes′ = expand_scalar(T, f)
    for l in 0:lmax, m in -l:l
        @test abs(modes[(l,m)] - modes′[(l,m)]) <= 10*atol
    end
end

@testset "evaluate coordinates" begin
    T = Float64
    atol = sqrt(eps(T))
    x(θ,ϕ) = sin(θ) * cos(ϕ)
    y(θ,ϕ) = sin(θ) * sin(ϕ)
    z(θ,ϕ) = cos(θ)
    mx = ScalarModes{T}()
    my = ScalarModes{T}()
    mz = ScalarModes{T}()
    for l in 0:lmax, m in -l:l
        mx[(l,m)] = 0
        my[(l,m)] = 0
        mz[(l,m)] = 0
    end
    mx[(1,-1)] = sqrt(2π/3)
    mx[(1,1)] = -sqrt(2π/3)
    my[(1,-1)] = sqrt(2π/3)*im
    my[(1,1)] = sqrt(2π/3)*im
    mz[(1,0)] = sqrt(4π/3)
    x′(θ,ϕ) = eval_scalar(mx, θ,ϕ)
    y′(θ,ϕ) = eval_scalar(my, θ,ϕ)
    z′(θ,ϕ) = eval_scalar(mz, θ,ϕ)
    for θ in range(0, T(π), length=21), ϕ in range(0, 2*T(π), length=41)
        @test abs(eval_scalar(mx, θ,ϕ) - x(θ,ϕ)) <= 10*atol
        @test abs(eval_scalar(my, θ,ϕ) - y(θ,ϕ)) <= 10*atol
        @test abs(eval_scalar(mz, θ,ϕ) - z(θ,ϕ)) <= 10*atol
    end
end

@testset "expand_grad" begin
    T = Float64
    atol = sqrt(eps(T))
    for l in 1:lmax, m in -l:l
        f(θ,ϕ) = gradYlm(l,m,θ,ϕ)
        modes = expand_grad(T, f)
        for l′=1:lmax, m′=-l′:l′
            δ = (l,m)==(l′,m′)
            @test abs(modes[(l′,m′)] - δ) <= 10*atol
        end
    end
end

@testset "eval_grad" begin
    T = Float64
    atol = sqrt(eps(T))
    # Choose function
    modes = Dict{NTuple{2,Int}, Complex{T}}()
    for l in 1:lmax, m in -l:l
        modes[(l,m)] = rand(Complex{T})
    end
    f(θ,ϕ) = eval_grad(modes, θ,ϕ)
    # Expand into modes
    modes′ = expand_grad(T, f)
    for l in 1:lmax, m in -l:l
        @test abs(modes[(l,m)] - modes′[(l,m)]) <= 10*atol
    end
end

@testset "expand_curl" begin
    T = Float64
    atol = sqrt(eps(T))
    for l in 1:lmax, m in -l:l
        f(θ,ϕ) = curlYlm(l,m,θ,ϕ)
        modes = expand_curl(T, f)
        for l′=1:lmax, m′=-l′:l′
            δ = (l,m)==(l′,m′)
            @test abs(modes[(l′,m′)] - δ) <= 10*atol
        end
    end
end

@testset "eval_curl" begin
    T = Float64
    atol = sqrt(eps(T))
    # Choose function
    modes = Dict{NTuple{2,Int}, Complex{T}}()
    for l in 1:lmax, m in -l:l
        modes[(l,m)] = rand(Complex{T})
    end
    f(θ,ϕ) = eval_curl(modes, θ,ϕ)
    # Expand into modes
    modes′ = expand_curl(T, f)
    for l in 1:lmax, m in -l:l
        @test abs(modes[(l,m)] - modes′[(l,m)]) <= 10*atol
    end
end

@testset "gradient_scalar" begin
end

@testset "curl_scalar" begin
end

@testset "div_gradient" begin
end

@testset "div_curl" begin
end

using DivGradYlm

using Test



const BigRat = Rational{BigInt}
Base.eps(::Type{BigRat}) = BigRat(0)
Base.rand(::Type{BigRat}) = BigRat(rand(-1000:1000)) / 1000
Base.rand(::Type{Complex{BigRat}}) = Complex{BigRat}(rand(BigRat), rand(BigRat))



@testset "Basis function" begin

    T = Float64
    atol = sqrt(eps(T))

    @testset "Ylm" begin
        for l in 0:lmax, m in -l:l, l′=0:lmax, m′=-l′:l′
            I = sphere_dot(T,
                           (θ,ϕ) -> Ylm(l,m,θ,ϕ),
                           (θ,ϕ) -> Ylm(l′,m′,θ,ϕ))
            δ = (l,m)==(l′,m′)
            @test abs(I - δ) <= 10*atol
        end
    end

    @testset "gradYlm" begin
        for l in 0:lmax, m in -l:l, l′=0:lmax, m′=-l′:l′
            I = sphere_vdot(T,
                            (θ,ϕ) -> gradYlm(l,m,θ,ϕ),
                            (θ,ϕ) -> gradYlm(l′,m′,θ,ϕ))
            δ = (l,m)==(l′,m′)
            @test abs(I - l*(l+1)*δ) <= 10*atol
        end
    end

    @testset "curlYlm" begin
        for l in 0:lmax, m in -l:l, l′=0:lmax, m′=-l′:l′
            I = sphere_vdot(T,
                            (θ,ϕ) -> curlYlm(l,m,θ,ϕ),
                            (θ,ϕ) -> curlYlm(l′,m′,θ,ϕ))
            δ = (l,m)==(l′,m′)
            @test abs(I - l*(l+1)*δ) <= 10*atol
        end
    end

    @testset "grad/curlYlm" begin
        for l in 0:lmax, m in -l:l, l′=0:lmax, m′=-l′:l′
            I = sphere_vdot(T,
                            (θ,ϕ) -> gradYlm(l,m,θ,ϕ),
                            (θ,ϕ) -> curlYlm(l′,m′,θ,ϕ))
            @test abs(I) <= 10*atol
        end
    end

    @testset "traceYlm" begin
        for l in 0:lmax, m in -l:l, l′=0:lmax, m′=-l′:l′
            I = sphere_tdot(T,
                            (θ,ϕ) -> traceYlm(l,m,θ,ϕ),
                            (θ,ϕ) -> traceYlm(l′,m′,θ,ϕ))
            δ = (l,m)==(l′,m′)
            # Note: The paper says I = -2δ
            @test abs(I - 2δ) <= 10*atol
        end
    end

    @testset "gradgradYlm" begin
        for l in 0:lmax, m in -l:l, l′=0:lmax, m′=-l′:l′
            I = sphere_tdot(T,
                            (θ,ϕ) -> gradgradYlm(l,m,θ,ϕ),
                            (θ,ϕ) -> gradgradYlm(l′,m′,θ,ϕ))
            δ = (l,m)==(l′,m′)
            @test abs(I - l*(l+1) * (l*(l+1)÷2 - 1) * δ) <= 10*atol
        end
    end

end



@testset "Expanding and evaluating" begin

    T = Float64
    atol = sqrt(eps(T))

    @testset "expand_scalar" begin
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

    x(θ,ϕ) = sin(θ) * cos(ϕ)
    y(θ,ϕ) = sin(θ) * sin(ϕ)
    z(θ,ϕ) = cos(θ)

    @testset "expand coordinates" begin
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
        # Choose function
        modes = rand(ScalarModes{T})
        f(θ,ϕ) = eval_scalar(modes, θ,ϕ)
        # Expand into modes
        modes′ = expand_scalar(T, f)
        for l in 0:lmax, m in -l:l
            @test abs(modes[(l,m)] - modes′[(l,m)]) <= 10*atol
        end
    end

    @testset "evaluate coordinates" begin
        mx = zero(ScalarModes{T})
        my = zero(ScalarModes{T})
        mz = zero(ScalarModes{T})
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
        # Choose function
        modes = rand(GradModes{T})
        f(θ,ϕ) = eval_grad(modes, θ,ϕ)
        # Expand into modes
        modes′ = expand_grad(T, f)
        for l in 1:lmax, m in -l:l
            @test abs(modes[(l,m)] - modes′[(l,m)]) <= 10*atol
        end
    end

    @testset "expand_curl" begin
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
        # Choose function
        modes = rand(CurlModes{T})
        f(θ,ϕ) = eval_curl(modes, θ,ϕ)
        # Expand into modes
        modes′ = expand_curl(T, f)
        for l in 1:lmax, m in -l:l
            @test abs(modes[(l,m)] - modes′[(l,m)]) <= 10*atol
        end
    end

end



@testset "Derivative operators" begin

    T = BigRat

    @testset "gradient_scalar" begin
        # Choose function
        smodes = rand(ScalarModes{T})
        gmodes = grad_scalar(smodes)
        gmodes::GradModes{T}
    end

    @testset "div_scalar" begin
        # Choose function
        smodes = rand(ScalarModes{T})
        dmodes = div_scalar(smodes)
        dmodes::ScalarModes{T}
    end

    @testset "div_grad" begin
        # Choose function
        gmodes = rand(GradModes{T})
        smodes = div_grad(gmodes)
        smodes::ScalarModes{T}
    end

    @testset "curl_scalar" begin
        # Choose function
        smodes = rand(ScalarModes{T})
        cmodes = curl_scalar(smodes)
        cmodes::CurlModes{T}
    end

    @testset "curl_grad" begin
        # Choose function
        gmodes = rand(GradModes{T})
        cmodes = curl_grad(gmodes)
        cmodes::CurlModes{T}
    end

    @testset "curl_curl" begin
        # Choose function
        cmodes = rand(CurlModes{T})
        smodes, gmodes = curl_curl(cmodes)
        smodes::ScalarModes{T}
        gmodes::GradModes{T}
    end

    @testset "laplace_scalar" begin
        # Choose function
        smodes = rand(ScalarModes{T})
        lmodes = laplace_scalar(smodes)
        lmodes::ScalarModes{T}
        for l in 0:lmax, m in -l:l
            @test lmodes[(l,m)] == -l*(l+1) * smodes[(l,m)]
        end
    end

end

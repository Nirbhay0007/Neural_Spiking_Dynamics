using Lux, SciMLSensitivity, Optimization, OptimizationOptimisers, Statistics, Random, ComponentArrays, Zygote, DifferentialEquations
using CUDA
CUDA.allowscalar(false) # Catch scalar indexing

gpu_dev = gpu_device()
cpu_dev = cpu_device()

t_train_cpu = Float32.(collect(0:0.1:50.0))
z_train_cpu = rand(Float32, 4, length(t_train_cpu))
t_train = t_train_cpu |> gpu_dev
z_train = z_train_cpu |> gpu_dev
scale_factors = rand(Float32, 4, 1) |> gpu_dev

nn = Lux.Chain(Lux.Dense(5 => 64, Lux.tanh), Lux.LayerNorm(64), Lux.Dense(64 => 64, Lux.tanh), Lux.LayerNorm(64), Lux.Dense(64 => 4))
rng = Random.default_rng()
ps, st = Lux.setup(rng, nn)

ps_gpu = ps |> gpu_dev
st_gpu = st |> gpu_dev
p_flat, reconstruct = Optimisers.destructure(ps_gpu)
p_initial = Float32.(p_flat)

α_m(V) = 0.1f0 .* (V .+ 40.0f0) ./ (1.0f0 .- exp.(-(V .+ 40.0f0) ./ 10.0f0))
β_m(V) = 4.0f0 .* exp.(-(V .+ 65.0f0) ./ 18.0f0)
α_h(V) = 0.07f0 .* exp.(-(V .+ 65.0f0) ./ 20.0f0)
β_h(V) = 1.0f0 ./ (1.0f0 .+ exp.(-(V .+ 35.0f0) ./ 10.0f0))
α_n(V) = 0.01f0 .* (V .+ 55.0f0) ./ (1.0f0 .- exp.(-(V .+ 55.0f0) ./ 10.0f0))
β_n(V) = 0.125f0 .* exp.(-(V .+ 65.0f0) ./ 80.0f0)

function hh_equations(u)
    V = u[1:1, :] 
    n = u[2:2, :]
    m = u[3:3, :]
    h = u[4:4, :]
    I_ext = 10.0f0; g_Na = 120.0f0; g_K = 36.0f0; g_L = 0.3f0; E_Na = 50.0f0; E_K = -77.0f0; E_L = -54.4f0; C_m = 1.0f0
    I_Na = g_Na .* (m.^3) .* h .* (V .- E_Na)
    I_K  = g_K .* (n.^4) .* (V .- E_K)
    I_L  = g_L .* (V .- E_L)
    dV = (I_ext .- I_Na .- I_K .- I_L) ./ C_m
    dn = α_n.(V) .* (1.0f0 .- n) .- β_n.(V) .* n
    dm = α_m.(V) .* (1.0f0 .- m) .- β_m.(V) .* m
    dh = α_h.(V) .* (1.0f0 .- h) .- β_h.(V) .* h
    return vcat(dV, dn, dm, dh)
end

function pyhysics_loss(nn, state_data, time_data, θ, scale_factors, reconstruct)
    ps_current = reconstruct(θ)
    nn_inputs = vcat(state_data, reshape(time_data, 1, :))
    pred_derivs, _ = nn(nn_inputs, ps_current, st_gpu)
    true_dervis = hh_equations(state_data)
    residuals = (pred_derivs .- true_dervis) ./ scale_factors
    return mean(residuals.^2)
end

function neural_dynamics(u, p, t)
    ps_structured = reconstruct(p)
    t_arr = fill(t, 1) |> gpu_dev
    raw_input = vcat(u, t_arr) 
    input_2d = reshape(raw_input, :, 1)
    dudt_2d, _ = nn(input_2d, ps_structured, st_gpu) 
    return vec(dudt_2d)
end

u0_gpu = Float32[-65.0f0, 0.05f0, 0.6f0, 0.32f0] |> gpu_dev
tspan = (0.0f0, 50.0f0)
node_prob = ODEProblem(neural_dynamics, u0_gpu, tspan, p_initial)

function predict(θ)
    new_prob = remake(node_prob, p=θ)
    return solve(new_prob, Heun(), saveat=t_train, reltol=1e-3, abstol=1e-3, dt=0.1f0, sensealg=InterpolatingAdjoint())
end

function total_loss(θ, _)
    sol = predict(θ)
    if sol.retcode != ReturnCode.Success
        return Inf32
    end
    loss_data = mean(abs2, Array(sol) .- z_train)
    loss_phys = pyhysics_loss(nn, z_train, t_train, θ, scale_factors, reconstruct)
    return loss_data + 1.5f0 * loss_phys
end

println("Starting test...")
@time total_loss(p_initial, nothing)

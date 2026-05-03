using Lux, SciMLSensitivity, Optimization, OptimizationOptimisers, Statistics, Random, ComponentArrays, StaticArrays, Zygote, DifferentialEquations
using CUDA

println("CUDA Available: ", CUDA.functional())

# Setup devices
gpu_dev = gpu_device()
cpu_dev = cpu_device()

# Parameters and Initial Condition
p = Float32[
    10.0f0,  120.0f0, 36.0f0,  0.3f0,   
    50.0f0, -77.0f0, -54.4f0,  1.0f0    
]
u0 = Float32[-65.0f0, 0.05f0, 0.6f0, 0.32f0]
tspan = (0.0f0, 50.0f0)

α_m(V) = 0.1f0 .* (V .+ 40.0f0) ./ (1.0f0 .- exp.(-(V .+ 40.0f0) ./ 10.0f0))
β_m(V) = 4.0f0 .* exp.(-(V .+ 65.0f0) ./ 18.0f0)
α_h(V) = 0.07f0 .* exp.(-(V .+ 65.0f0) ./ 20.0f0)
β_h(V) = 1.0f0 ./ (1.0f0 .+ exp.(-(V .+ 35.0f0) ./ 10.0f0))
α_n(V) = 0.01f0 .* (V .+ 55.0f0) ./ (1.0f0 .- exp.(-(V .+ 55.0f0) ./ 10.0f0))
β_n(V) = 0.125f0 .* exp.(-(V .+ 65.0f0) ./ 80.0f0)

function hodgkin_huxley!(du, u, p, t)
    V, m, h, n = u
    I_ext, g_Na, g_K, g_L, E_Na, E_K, E_L, C = p

    # 1. Calculate ionic currents
    I_Na = g_Na .* (m.^3) .* h .* (V .- E_Na)
    I_K  = g_K .* (n.^4) .* (V .- E_K)
    I_L  = g_L .* (V .- E_L)

    # 2. Membrane potential differential equation
    du[1] = (I_ext .- I_Na .- I_K .- I_L) ./ C

    # 3. Gating variable differential equations (dx/dt = α(1-x) - βx)
    du[2] = α_m(V) * (1f0 - m) - β_m(V) * m
    du[3] = α_h(V) * (1f0 - h) - β_h(V) * h
    du[4] = α_n(V) * (1f0 - n) - β_n(V) * n
end

rng = Random.default_rng()
nn = Lux.Chain(
    Lux.Dense(5 => 64, Lux.tanh),
    Lux.LayerNorm(64),
    Lux.Dense(64 => 64, Lux.tanh),
    Lux.LayerNorm(64),
    Lux.Dense(64 => 4)
)
ps, st = Lux.setup(rng, nn)

# Transfer to GPU
ps_gpu = ps |> gpu_dev
st_gpu = st |> gpu_dev

p_flat_gpu, reconstruct_gpu = Optimisers.destructure(ps_gpu)

# ODE state must be on GPU
u0_gpu = u0 |> gpu_dev
p_gpu = p |> gpu_dev

function neural_dynamics(u, p, t)
    ps_structured = reconstruct_gpu(p)
    # vcat and reshape on GPU
    # t is scalar during ODE solve, we need to wrap it into an array
    t_arr = fill(t, 1) |> gpu_dev
    raw_input = vcat(u, t_arr)
    input_2d = reshape(raw_input, :, 1)
    dudt_2d, _ = nn(input_2d, ps_structured, st_gpu) 
    return vec(dudt_2d)
end

node_prob = ODEProblem(neural_dynamics, u0_gpu, tspan, p_flat_gpu)

# Ground truth data generation (CPU is fine here, it's just once)
true_prob = ODEProblem(hodgkin_huxley!, u0, tspan, p)
true_sol = solve(true_prob, Heun(), dt=0.1f0, saveat=0.5f0)
t_train_cpu = Float32.(true_sol.t)
z_train_cpu = Float32.(Array(true_sol))

t_train = t_train_cpu |> gpu_dev
z_train = z_train_cpu |> gpu_dev

function calculate_scale_factors(z_train, t_train)
    dt = diff(t_train)
    dz = diff(z_train, dims=2)
    derivatives = dz ./ dt'
    s_factors = std(derivatives, dims=2)
    return max.(s_factors, 1e-5f0)
end
scale_factors_gpu = calculate_scale_factors(z_train_cpu, t_train_cpu) |> gpu_dev

function predict(θ)
    new_prob = remake(node_prob, p=θ)
    return solve(new_prob, Heun(), 
                 saveat=t_train, 
                 reltol=1e-3, abstol=1e-3, dt=0.1f0,
                 sensealg=InterpolatingAdjoint())
end

function physics_loss(θ, state_data, time_data, scale_factors)
    ps_current = reconstruct_gpu(θ)
    nn_inputs = vcat(state_data, reshape(time_data, 1, :))
    pred_derivs, _ = nn(nn_inputs, ps_current, st_gpu)

    # Array broadcasting on GPU is better than in-place mutation for a batch!
    V = state_data[1:1, :]
    m = state_data[2:2, :]
    h = state_data[3:3, :]
    n = state_data[4:4, :]
    
    I_ext, g_Na, g_K, g_L, E_Na, E_K, E_L, C = p_gpu[1], p_gpu[2], p_gpu[3], p_gpu[4], p_gpu[5], p_gpu[6], p_gpu[7], p_gpu[8]
    
    I_Na = g_Na .* (m.^3) .* h .* (V .- E_Na)
    I_K  = g_K .* (n.^4) .* (V .- E_K)
    I_L  = g_L .* (V .- E_L)
    
    dV = (I_ext .- I_Na .- I_K .- I_L) ./ C
    dm = α_m.(V) .* (1f0 .- m) .- β_m.(V) .* m
    dh = α_h.(V) .* (1f0 .- h) .- β_h.(V) .* h
    dn = α_n.(V) .* (1f0 .- n) .- β_n.(V) .* n
    
    true_derivs = vcat(dV, dm, dh, dn)
    
    residuals = (pred_derivs .- true_derivs) ./ scale_factors
    return mean(abs2, residuals)
end

function total_loss(θ, _)
    sol = predict(θ)
    if sol.retcode != ReturnCode.Success
        return Inf32
    end
    
    loss_data = mean(abs2, Array(sol) .- z_train)
    loss_phys = physics_loss(θ, z_train, t_train, scale_factors_gpu)
    
    λ = 1.5f0
    return loss_data + λ * loss_phys
end

loss_history = Float32[]
callback_fn = function (θ, loss_val)
    push!(loss_history, loss_val)
    if length(loss_history) % 1 == 0
        println("Iteration: $(length(loss_history)) | Loss: $(loss_val)")
    end
    return false
end

optf = OptimizationFunction(total_loss, Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p_flat_gpu)

println("Starting Training on GPU...")
result = solve(optprob, Adam(0.01), maxiters = 5, callback = callback_fn)
println("Training Complete!")

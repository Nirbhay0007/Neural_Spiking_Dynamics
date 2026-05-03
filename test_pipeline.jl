using Lux, SciMLSensitivity, Optimization, OptimizationOptimisers, Statistics, Random, ComponentArrays, StaticArrays, Zygote, DifferentialEquations
using CUDA

p = Float32[
    10.0f0,  # I_ext
    120.0f0, # g_Na
    36.0f0,  # g_K
    0.3f0,   # g_L
    50.0f0,  # E_Na
    -77.0f0, # E_K
    -54.4f0, # E_L
    1.0f0    # C
]
u0 = Float32[-65.0f0, 0.05f0, 0.6f0, 0.32f0]
tspan = (0.0f0, 50.0f0)

α_m(V) = 0.1f0 * (V + 40.0f0) / (1.0f0 - exp(-(V + 40.0f0) / 10.0f0))
β_m(V) = 4.0f0 * exp(-(V + 65.0f0) / 18.0f0)
α_h(V) = 0.07f0 * exp(-(V + 65.0f0) / 20.0f0)
β_h(V) = 1.0f0 / (1.0f0 + exp(-(V + 35.0f0) / 10.0f0))
α_n(V) = 0.01f0 * (V + 55.0f0) / (1.0f0 - exp(-(V + 55.0f0) / 10.0f0))
β_n(V) = 0.125f0 * exp(-(V + 65.0f0) / 80.0f0)

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
    du[2] = α_m.(V) .* (1f0 .- m) .- β_m.(V) .* m
    du[3] = α_h.(V) .* (1f0 .- h) .- β_h.(V) .* h
    du[4] = α_n.(V) .* (1f0 .- n) .- β_n.(V) .* n
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
p_flat, reconstruct = Optimisers.destructure(ps)
p_initial = Float32.(p_flat)

function neural_dynamics(u, p, t)
    ps_structured = reconstruct(p)
    raw_input = vcat(u, t) 
    input_2d = reshape(raw_input, :, 1)
    dudt_2d, _ = nn(input_2d, ps_structured, st) 
    return vec(dudt_2d)
end
node_prob = ODEProblem(neural_dynamics, u0, tspan, p_initial)

# Generate ground truth training data using the optimized HH model
true_prob = ODEProblem(hodgkin_huxley!, u0, tspan, p)
true_sol = solve(true_prob, Heun(), dt=0.1f0, saveat=0.5f0)
t_train = Float32.(true_sol.t)
z_train = Float32.(Array(true_sol))

function calculate_scale_factors(z_train, t_train)
    dt = diff(t_train)
    dz = diff(z_train, dims=2)
    derivatives = dz ./ dt'
    s_factors = std(derivatives, dims=2)
    # Fallback to avoid division by zero
    return max.(s_factors, 1e-5f0)
end
scale_factors = calculate_scale_factors(z_train, t_train)

function predict(θ)
    new_prob = remake(node_prob, p=θ)
    return solve(new_prob, Heun(), 
                 saveat=t_train, 
                 reltol=1e-3, abstol=1e-3, dt=0.1f0,
                 sensealg=InterpolatingAdjoint())
end

function physics_loss(θ, state_data, time_data, scale_factors)
    ps_current = reconstruct(θ)
    nn_inputs = vcat(state_data, time_data')
    pred_derivs, _ = nn(nn_inputs, ps_current, st)

    true_derivs = zeros(Float32, size(state_data))
    for i in 1:size(state_data, 2)
        # In-place derivatives calculation
        du_temp = MVector{4, Float32}(0f0,0f0,0f0,0f0)
        hodgkin_huxley!(du_temp, state_data[:, i], p, time_data[i])
        true_derivs[:, i] .= du_temp
    end

    residuals = (pred_derivs .- true_derivs) ./ scale_factors
    return mean(residuals.^2)
end

function total_loss(θ, _)
    sol = predict(θ)
    if sol.retcode != ReturnCode.Success
        return Inf32
    end
    
    # Data loss
    loss_data = mean(abs2, Array(sol) .- z_train)
    
    # Physics residual loss
    loss_phys = physics_loss(θ, z_train, t_train, scale_factors)
    
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
optprob = OptimizationProblem(optf, p_initial)

println("Starting Training...")
# Train for 5 epochs to verify loss convergence quickly
result = solve(optprob, Adam(0.01), maxiters = 5, callback = callback_fn)
println("Training Complete!")

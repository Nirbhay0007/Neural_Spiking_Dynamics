using Turing, Distributions

# 1. Define a simple model
@model function demo_model(y)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    for i in eachindex(y)
        y[i] ~ Normal(m, sqrt(s))
    end
end

data = randn(100)
model_inst = demo_model(data)

# 2. EXTREMELY FAST VERSION (Multi-threaded)
# This will launch 8 chains simultaneously. 
# Watch your CPU usage spike in Task Manager!
println("Starting parallel sampling...")
@time chain = sample(model_inst, NUTS(), MCMCThreads(), 1000, 8)

println("Sampling Complete.")


using Lux, DiffEqFlux, DifferentialEquations, CUDA, Random

# 1. Setup Device
device = CUDA.functional() ? gpu_device() : cpu_device()
println("Training on: ", device)

# 2. Define a small Neural ODE
CUDA.functional()
rng = Random.default_rng()
nn = Chain(Dense(2 => 10, tanh), Dense(10 => 2))
ps, st = Lux.setup(rng, nn) |> device  # MOVE TO GPU HERE

n_ode = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(), saveat=0.1f0)

# 3. Example Forward Pass
# This runs on the GPU if 'device' is set to GPU
x0 = randn(Float32, 2, 1) |> device
println("Running GPU Forward Pass...")
@time prediction = n_ode(x0, ps, st)

# 4. Use Adjoint for fast gradients (The "Secret Sauce")
using SciMLSensitivity
# This solves the ODE using adjoint sensitivity, which is much faster for training
sol = solve(ODEProblem((u, p, t) -> nn(u, p, st)[1], x0, (0.0f0, 1.0f0), ps),
    Tsit5(), sensealg=InterpolatingAdjoint())
# -------------------------
using Base.Threads
using CUDA
using BenchmarkTools

# A simple math task (squaring numbers)
data = rand(Float32, 10^7)

# --- METHOD A: THE "SLOW" WAY (Single Thread) ---
function slow_way(x)
    return x .^ 2
end

# --- METHOD B: THE "FAST" CPU WAY (Multi-threaded) ---
# This uses the "auto" threads you set in your VS Code settings
function fast_cpu(x)
    out = similar(x)
    @threads for i in eachindex(x)
        out[i] = x[i]^2
    end
    return out
end

# --- METHOD C: THE "EXTREME" WAY (GPU) ---
# This moves data to your NVIDIA card for instant calculation
function extreme_gpu(x)
    x_gpu = CuArray(x) # Move to GPU
    return x_gpu .^ 2  # Calculate on GPU cores
end

println("Testing Slow CPU...")
@btime slow_way($data)

println("Testing Fast Multi-threaded CPU...")
@btime fast_cpu($data)

if CUDA.functional()
    println("Testing Extreme GPU...")
    @btime extreme_gpu($data)
end
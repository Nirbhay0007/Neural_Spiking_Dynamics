using SciMLSensitivity, DifferentialEquations, ModelingToolkit, Optimization, OptimizationOptimisers
using Lux, Random, ComponentArrays, CUDA, Zygote
using Plots, JLD2

# 1. SETUP & DATA (Move to GPU)
# ------------------------------------------------
@load "Data/synthetic_data/noise_0_hh_2d_model.jld" V
t_train = 0.0f0:0.1f0:50.0f0

# Move data to GPU
V_gpu = cu(Float32.(V))
t_train_gpu = cu(t_train)

# 2. DEFINE MODEL WITH LUX (The Modern Way)
# ------------------------------------------------
# Lux models are explicit. We don't 'destructure' them.
# We define the structure once.
nn = Lux.Chain(
    Lux.Dense(1, 16, tanh),
    Lux.Dense(16, 1)
)

# Initialize parameters (p) and state (st)
rng = Random.default_rng()
p_init, st = Lux.setup(rng, nn)

# CRITICAL STABILITY FIX
# Set final layer weights to zero to prevent initial explosion
p_init.layer_2.weight .= 0.0f0
p_init.layer_2.bias   .= 0.0f0

# Move parameters to GPU
p_gpu = cu(ComponentArray(p_init))
st_gpu = cu(st) # State also needs to be on GPU

# External stimulus function (simple square pulse). Replace with your experiment's stimulus as needed.
function Stimulus(t)
    return (t >= 10.0f0 && t < 11.0f0) ? 20.0f0 : 0.0f0
end

# 3. UDE FUNCTION (Optimized for GPU)
# ------------------------------------------------
# We pass the Lux model structure 'nn' and state 'st' via a closure or global
function hodgkin_huxley_UDE!(du, u, p, t, nn, st)
    # Avoid scalar indexing on a CuArray `u` (disallowed on GPU).
    # Copy the small state vector to CPU and work with it instead.
    u_local = Array(u)
    V, n = u_local
    
    # LUX FORWARD PASS
    # Run the NN on GPU (parameters `p`/`st` are on GPU) but keep physics on CPU scalars.
    _V_input_gpu = cu(Float32[V])
    pred_gpu, _ = nn(_V_input_gpu, p, st)
    pred_cpu = Array(pred_gpu)
    pred_I_Na = pred_cpu[1]
    
    # Physics calculations on CPU scalars
    I_ext = Stimulus(t)
    I_K = g_K * n^4 * (V - E_K)
    I_L = g_L * (V - E_L)
    du_local1 = (I_ext - (pred_I_Na + I_K + I_L) / Cm)
    du_local2 = (n_inf(V) - n) / tau_n(V)
    
    # Transfer derivatives back to GPU `du`
    du .= cu(Float32[du_local1, du_local2])
end

# 4. WRAPPER FOR SCIML
# ------------------------------------------------
# DifferentialEquations expects (du, u, p, t)
# We wrap the Lux model and state into the function
ude_dynamics!(du, u, p, t) = hodgkin_huxley_UDE!(du, u, p, t, nn, st_gpu)

u0_gpu = cu([-65.0f0, 0.317f0])
prob_nn = ODEProblem(ude_dynamics!, u0_gpu, (0.0f0, 50.0f0), p_gpu)

# 5. SOLVE (Using a GPU-friendly solver)
# ------------------------------------------------
function loss(p)
    # GBS (Grossmann-Bulirsch-Stoer) is often robust for simple problems
    # Tsit5 is standard. 
    pred = solve(prob_nn, Tsit5(), p=p, saveat=t_train_gpu, 
                 reltol=1e-4, abstol=1e-4) # Lower tolerance for Float32
    
    if pred.retcode != :Success
        return 1f6
    end
    
    # Vectorized GPU loss calculation (compute on CPU to avoid scalar GPU indexing)
    pred_arr = Array(pred)  # materialize solution on CPU
    loss_val = sum(abs2, pred_arr[1,:] .- Array(V_gpu))
    return loss_val
end

# ... Optimization follows ...
optf = Optimization.OptimizationFunction((x, p) -> loss(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, p_gpu) # Pass GPU parameters

res = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), maxiters=10)
println(res)

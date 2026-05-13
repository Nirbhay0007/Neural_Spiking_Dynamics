# Cell 1: Setup and Continuous Domain Prep
using Lux, SciMLSensitivity, Optimization, OptimizationOptimisers, Statistics, Random, ComponentArrays, Zygote

using CSV, DataFrames

# Define the file path (using forward slashes for Windows compatibility)
file_path = "c:/Users/ADMIN/Downloads/Neural_Spiking_Dynamics/notebooks/1_data_generation/single_spike_noisy_data.csv"

# Read the CSV into a DataFrame called HH_data
HH_data = CSV.read(file_path, DataFrame)


df_ordered = DataFrame(
    timestamp=HH_data.timestamp,
    V=HH_data.V,
    n=HH_data.h,
    m=HH_data.m,
    h=HH_data.n
)
# Display the first few rows to verify
first(df_ordered, 5)

# Now you can run your existing code:
# df_ordered = HH_data[:, [:timestamp, :V, :n, :m, :h]]

end_timestamp = df_ordered.timestamp[end]
df_ordered.timestamp[end]


# 2. Move your training data to the GPU
t_train = Float32.(df_ordered.timestamp)
z_train = Float32.(Matrix(df_ordered[:, [:V, :n, :m, :h]])')

rng = Random.default_rng()
Random.seed!(rng, 42)

# Generate Collocation Points (e.g., 2000 random points between 0 and 50 ms)
# The network will use AutoDiff at these points to enforce the PDE
N_colloc = 2000
t_min, t_max = minimum(t_train), maximum(t_train)
t_colloc = Float32.(rand(rng, 1, N_colloc) .* (t_max - t_min) .+ t_min)

# Cell 2: Mesh-Free Continuous Architecture

# Build the standard PINN MLP
# Inputs: 1 (Continuous Time, t)
# Outputs: 4 (V, n, m, h)
pinn_model = Lux.Chain(
    Lux.Dense(1 => 32, sin),    # Using 'sin' instead of tanh helps with stiff/spiky data
    Lux.Dense(32 => 64, sin),
    Lux.Dense(64 => 64, sin),
    Lux.Dense(64 => 32, sin),
    Lux.Dense(32 => 4)
)

# Initialize network weights and states
ps, st = Lux.setup(rng, pinn_model)
p_nn = ComponentArray(ps)


# Cell 3: Data and AutoDiff PDE Loss

# (Assumes your standard hh_equations(u) function is loaded)

function physics_loss_continuous(t_c, p)
    # 1. Forward pass on collocation points
    u_pred, _ = pinn_model(t_c, p, st)

    # 2. AUTOMATIC DIFFERENTIATION (The core PINN mechanic)
    # We want du/dt for all 4 states. 
    # A standard Zygote trick for batch 1D gradients: taking the gradient of the 
    # sum of outputs w.r.t inputs gives the exact element-wise derivatives.
    du_dt = vcat([
        Zygote.gradient(t -> sum(pinn_model(t, p, st)[1][i, :]), t_c)[1]
        for i in 1:4
    ]...)

    # 3. Calculate true HH dynamics at these predicted states
    true_dynamics = hh_equations(u_pred)

    # 4. PDE Residual: The difference between AutoDiff derivative and Physical derivative
    # (Assuming scale_factors is available from your previous scaling code)
    residuals = (du_dt .- true_dynamics) ./ scale_factors

    return mean(abs2, residuals)
end

function total_loss(p, _)
    # --- A. Data Loss ---
    # Enforce that the network matches our known sensor readings
    u_data_pred, _ = pinn_model(t_data, p, st)
    loss_data = mean(abs2, u_data_pred .- z_data)

    # --- B. Physics Loss ---
    # Enforce that the network obeys the PDE at random collocation points
    loss_phys = physics_loss_continuous(t_colloc, p)

    # Balance the two losses (often requires heavy tuning in standard PINNs)
    λ = 1.0f0

    return loss_data + λ * loss_phys
end


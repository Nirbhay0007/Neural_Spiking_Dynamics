# CELL 0
using Flux, DifferentialEquations, SciMLSensitivity, Optimization, OptimizationOptimisers, Statistics,DataFrames,CSV,Plots,Random

# CELL 1
data = "notebooks/1_data_generation/0_noise/HH_4D_data.csv"

# CELL 2
file_path = "c:/Users/nirbh/Neural_Spiking_Dynamics/notebooks/1_data_generation/0_noise/HH_4D_data.csv"
HH_data = CSV.read(file_path, DataFrame)

# CELL 3
df_ordered = HH_data[:, [:t, :V, :n, :m, :h]]

# CELL 4
t_train =  Float32.(df_ordered.t)
z_train = Float32.(Matrix(df_ordered[:, [:V, :n, :m, :h]])')

# CELL 5
using Plots
theme(:dark)
# Create 4 subplots in a 2x2 grid
p1 = plot(df_ordered.t, df_ordered.V, title="Voltage (V)", ylabel="mV")
p2 = plot(df_ordered.t, df_ordered.n, title="n-gate", color=:green)
p3 = plot(df_ordered.t, df_ordered.m, title="m-gate", color=:red)
p4 = plot(df_ordered.t, df_ordered.h, title="h-gate", color=:cyan)

plot(p1, p2, p3, p4, layout=(2, 2), legend=false)


# CELL 6
rng = Random.default_rng()


# CELL 7
function calculate_scale_factors(z_train,t_train)
          # det the time step (dt)
          dt = diff(t_train) # ( t_i+1 - t_i )

          dz = diff(z_train, dims=2)
          # calculate derivatives ( Finite difference)
          derivatives = dz ./ dt'
          ## result is a 4-element vector: [s_v, s_n, s_m, s_h]
          s_factors = std(derivatives, dims=2)
          


          return s_factors # [s_v, s_n, s_m, s_h]

end

# CELL 8
calculate_scale_factors(z_train,t_train)

# CELL 9
# 3-Layer MLP with LayerNorm
# Input: [v, n, m, h, t] (5 dimensions)
# Output: [dv, dn, dm, dh] (4 dimensions)


nn = Chain(
    Dense(5 => 64, tanh),
    LayerNorm(64),
    Dense(64 => 64, tanh),
    LayerNorm(64),
    Dense(64 => 4)
)





# CELL 10
function pyhysics_loss(nn, state_data, time_data, scale_factors)
          # get nn predicted derivatevies : dz/dt = f_0(z,t)
          # Note : state_data should include [V,n,m,h,t]
          pred_derivs = nn(state_data)

          # calculate the HH derivatevies using the equations in section
          true_dervis = hh_equations(state_data)


          #compute scale-aware residual
          residuals = (pred_derivs.-true_dervis)./scale_factors
          
          return mean(residuals.^2)
end

# CELL 11
# Voltage-dependent rate functions for the gating variables
# V is in millivolts (mV). These specific forms are tuned for a resting potential around -65mV.

α_m(V) = 0.1 * (V + 40.0) / (1.0 - exp(-(V + 40.0) / 10.0))
β_m(V) = 4.0 * exp(-(V + 65.0) / 18.0)

α_h(V) = 0.07 * exp(-(V + 65.0) / 20.0)
β_h(V) = 1.0 / (1.0 + exp(-(V + 35.0) / 10.0))

α_n(V) = 0.01 * (V + 55.0) / (1.0 - exp(-(V + 55.0) / 10.0))
β_n(V) = 0.125 * exp(-(V + 65.0) / 80.0)
# ------------------<>------------<>---------<>----

function hodgkin_huxley!(du, u, p, t)
    V, m, h, n = u
    I_ext, g_Na, g_K, g_L, E_Na, E_K, E_L, C = p

    # 1. Calculate ionic currents
    I_Na = g_Na * (m^3) * h * (V - E_Na)
    I_K  = g_K * (n^4) * (V - E_K)
    I_L  = g_L * (V - E_L)

    # 2. Membrane potential differential equation
    du[1] = (I_ext - I_Na - I_K - I_L) / C

    # 3. Gating variable differential equations (dx/dt = α(1-x) - βx)
    du[2] = α_m(V) * (1 - m) - β_m(V) * m
    du[3] = α_h(V) * (1 - h) - β_h(V) * h
    du[4] = α_n(V) * (1 - n) - β_n(V) * n
end

# ------------------<>------------<>---------<>----
# Standard HH Parameters
p = (
    I_ext = 10,  # External current (μA/cm²) - Try changing this to 2.0 or 20.0!
    120.0, # Max Sodium conductance (mS/cm²)
    36.0,  # Max Potassium conductance (mS/cm²)
    0.3,   # Leak conductance (mS/cm²)
    50.0,  # Sodium reversal potential (mV)
    -77.0, # Potassium reversal potential (mV)
    -54.4, # Leak reversal potential (mV)
    1.0    # Membrane capacitance (μF/cm²)
)

# Initial conditions: [V, m, h, n] roughly at resting state
u0 = Float32[-65.0, 0.05, 0.6, 0.32]
tspan = (0.0, 50.0) # Simulate for 50 illiseconds



# CELL 12
function total_loss(p, _)
          # Forward pass ( Data loss part)
          
          
          sol = solve(nn, Heun(),p=θ,reltol=1e-6, abstol=1e-6,sensealg=InterpolatingAdjoint())
          # Data loss
          loss_data = mean(abs2, Array(sol).-z_train)
          loss_phys = pyhysics_loss(nn, z_train, t_train, scale_factors)
          # combine with lambda
          λ = 1.5 
          return loss_data + λ * loss_phys

end

# CELL 13
# optimization 
optf  = OptimizationFunction(total_loss, Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p)
# Use Adam with a learning rate of 0.01
result = solve(optprob, Adam(0.001), maxiters = 100)


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


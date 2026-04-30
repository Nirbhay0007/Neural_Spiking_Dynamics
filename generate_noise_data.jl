using DifferentialEquations, DataFrames, CSV, Random, Statistics

# Voltage-dependent rate functions
α_m(V) = 0.1 * (V + 40.0) / (1.0 - exp(-(V + 40.0) / 10.0))
β_m(V) = 4.0 * exp(-(V + 65.0) / 18.0)
α_h(V) = 0.07 * exp(-(V + 65.0) / 20.0)
β_h(V) = 1.0 / (1.0 + exp(-(V + 35.0) / 10.0))
α_n(V) = 0.01 * (V + 55.0) / (1.0 - exp(-(V + 55.0) / 10.0))
β_n(V) = 0.125 * exp(-(V + 65.0) / 80.0)

function hodgkin_huxley!(du, u, p, t)
    V, m, h, n = u
    I_ext, g_Na, g_K, g_L, E_Na, E_K, E_L, C = p

    I_Na = g_Na * (m^3) * h * (V - E_Na)
    I_K  = g_K * (n^4) * (V - E_K)
    I_L  = g_L * (V - E_L)

    du[1] = (I_ext - I_Na - I_K - I_L) / C
    du[2] = α_m(V) * (1 - m) - β_m(V) * m
    du[3] = α_h(V) * (1 - h) - β_h(V) * h
    du[4] = α_n(V) * (1 - n) - β_n(V) * n
end

p = (
    I_ext = 10, g_Na = 120.0, g_K = 36.0, g_L = 0.3,
    E_Na = 50.0, E_K = -77.0, E_L = -54.4, C = 1.0
)
u0 = [-65.0, 0.05, 0.6, 0.32]
tspan = (0.0, 50.0)
prob = ODEProblem(hodgkin_huxley!, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6)

Random.seed!(42)

function apply_noise(clean_data, noise_pct)
    if noise_pct == 0
        return clean_data
    end
    sigma = std(clean_data) * (noise_pct / 100.0)
    return clean_data .+ randn(length(clean_data)) .* sigma
end

t_data = sol.t
V_clean = sol[1, :]
m_clean = sol[2, :]
h_clean = sol[3, :]
n_clean = sol[4, :]

out_base = "c:/Users/nirbh/Neural_Spiking_Dynamics/notebooks/1_data_generation"

for p in [0, 10, 20]
    df = DataFrame(
        t = t_data,
        V = apply_noise(V_clean, p),
        m = apply_noise(m_clean, p),
        h = apply_noise(h_clean, p),
        n = apply_noise(n_clean, p)
    )
    folder_path = joinpath(out_base, "$(p)_noise")
    mkpath(folder_path)
    CSV.write(joinpath(folder_path, "HH_4D_data.csv"), df)
    println("Saved $(p)% noise data to: $folder_path")
end
println("Done")

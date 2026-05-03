import json

path = r'C:\Users\ADMIN\Downloads\Neural_Spiking_Dynamics\Hudgin_Huxely_4D_model.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell to add dependencies
cell_deps = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "using Lux, SciMLSensitivity, Optimization, OptimizationOptimisers, Statistics, Random, ComponentArrays, StaticArrays, Zygote\n",
        "using CUDA"
    ]
}

# Cell to redefine parameters in Float32
cell_params = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "p = Float32[\n",
        "    10.0f0,  # I_ext\n",
        "    120.0f0, # g_Na\n",
        "    36.0f0,  # g_K\n",
        "    0.3f0,   # g_L\n",
        "    50.0f0,  # E_Na\n",
        "    -77.0f0, # E_K\n",
        "    -54.4f0, # E_L\n",
        "    1.0f0    # C\n",
        "]\n",
        "u0 = Float32[-65.0f0, 0.05f0, 0.6f0, 0.32f0]\n",
        "tspan = (0.0f0, 50.0f0)\n"
    ]
}

# Cell to define rate functions using Float32
cell_rates = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "α_m(V) = 0.1f0 * (V + 40.0f0) / (1.0f0 - exp(-(V + 40.0f0) / 10.0f0))\n",
        "β_m(V) = 4.0f0 * exp(-(V + 65.0f0) / 18.0f0)\n",
        "α_h(V) = 0.07f0 * exp(-(V + 65.0f0) / 20.0f0)\n",
        "β_h(V) = 1.0f0 / (1.0f0 + exp(-(V + 35.0f0) / 10.0f0))\n",
        "α_n(V) = 0.01f0 * (V + 55.0f0) / (1.0f0 - exp(-(V + 55.0f0) / 10.0f0))\n",
        "β_n(V) = 0.125f0 * exp(-(V + 65.0f0) / 80.0f0)\n"
    ]
}

# Cell to define hodgkin_huxley! with StaticArrays
cell_hh = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "function hodgkin_huxley!(du, u, p, t)\n",
        "    V, m, h, n = u\n",
        "    I_ext, g_Na, g_K, g_L, E_Na, E_K, E_L, C = p\n",
        "\n",
        "    # 1. Calculate ionic currents\n",
        "    I_Na = g_Na .* (m.^3) .* h .* (V .- E_Na)\n",
        "    I_K  = g_K .* (n.^4) .* (V .- E_K)\n",
        "    I_L  = g_L .* (V .- E_L)\n",
        "\n",
        "    # 2. Membrane potential differential equation\n",
        "    du[1] = (I_ext .- I_Na .- I_K .- I_L) ./ C\n",
        "\n",
        "    # 3. Gating variable differential equations (dx/dt = α(1-x) - βx)\n",
        "    du[2] = α_m.(V) .* (1f0 .- m) .- β_m.(V) .* m\n",
        "    du[3] = α_h.(V) .* (1f0 .- h) .- β_h.(V) .* h\n",
        "    du[4] = α_n.(V) .* (1f0 .- n) .- β_n.(V) .* n\n",
        "end\n"
    ]
}

# Add training loop and PI-NODE-SR configuration cells
cell_nn = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "rng = Random.default_rng()\n",
        "nn = Lux.Chain(\n",
        "    Lux.Dense(5 => 64, Lux.tanh),\n",
        "    Lux.LayerNorm(64),\n",
        "    Lux.Dense(64 => 64, Lux.tanh),\n",
        "    Lux.LayerNorm(64),\n",
        "    Lux.Dense(64 => 4)\n",
        ")\n",
        "ps, st = Lux.setup(rng, nn)\n",
        "p_flat, reconstruct = Optimisers.destructure(ps)\n",
        "p_initial = Float32.(p_flat)\n"
    ]
}

cell_neural_dyn = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "function neural_dynamics(u, p, t)\n",
        "    ps_structured = reconstruct(p)\n",
        "    raw_input = vcat(u, t) \n",
        "    input_2d = reshape(raw_input, :, 1)\n",
        "    dudt_2d, _ = nn(input_2d, ps_structured, st) \n",
        "    return vec(dudt_2d)\n",
        "end\n",
        "node_prob = ODEProblem(neural_dynamics, u0, tspan, p_initial)\n"
    ]
}

cell_training_data = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Generate ground truth training data using the optimized HH model\n",
        "true_prob = ODEProblem(hodgkin_huxley!, u0, tspan, p)\n",
        "true_sol = solve(true_prob, Heun(), dt=0.1f0, saveat=0.5f0)\n",
        "t_train = Float32.(true_sol.t)\n",
        "z_train = Float32.(Array(true_sol))\n"
    ]
}

cell_scale_factors = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "function calculate_scale_factors(z_train, t_train)\n",
        "    dt = diff(t_train)\n",
        "    dz = diff(z_train, dims=2)\n",
        "    derivatives = dz ./ dt'\n",
        "    s_factors = std(derivatives, dims=2)\n",
        "    # Fallback to avoid division by zero\n",
        "    return max.(s_factors, 1e-5f0)\n",
        "end\n",
        "scale_factors = calculate_scale_factors(z_train, t_train)\n"
    ]
}

cell_predict = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "function predict(θ)\n",
        "    new_prob = remake(node_prob, p=θ)\n",
        "    return solve(new_prob, Heun(), \n",
        "                 saveat=t_train, \n",
        "                 reltol=1e-3, abstol=1e-3, dt=0.1f0,\n",
        "                 sensealg=InterpolatingAdjoint())\n",
        "end\n"
    ]
}

cell_physics_loss = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "function physics_loss(θ, state_data, time_data, scale_factors)\n",
        "    ps_current = reconstruct(θ)\n",
        "    nn_inputs = vcat(state_data, time_data')\n",
        "    pred_derivs, _ = nn(nn_inputs, ps_current, st)\n",
        "\n",
        "    true_derivs = zeros(Float32, size(state_data))\n",
        "    for i in 1:size(state_data, 2)\n",
        "        # In-place derivatives calculation\n",
        "        du_temp = MVector{4, Float32}(0f0,0f0,0f0,0f0)\n",
        "        hodgkin_huxley!(du_temp, state_data[:, i], p, time_data[i])\n",
        "        true_derivs[:, i] .= du_temp\n",
        "    end\n",
        "\n",
        "    residuals = (pred_derivs .- true_derivs) ./ scale_factors\n",
        "    return mean(residuals.^2)\n",
        "end\n"
    ]
}

cell_total_loss = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "function total_loss(θ, _)\n",
        "    sol = predict(θ)\n",
        "    if sol.retcode != ReturnCode.Success\n",
        "        return Inf32\n",
        "    end\n",
        "    \n",
        "    # Data loss\n",
        "    loss_data = mean(abs2, Array(sol) .- z_train)\n",
        "    \n",
        "    # Physics residual loss\n",
        "    loss_phys = physics_loss(θ, z_train, t_train, scale_factors)\n",
        "    \n",
        "    λ = 1.5f0\n",
        "    return loss_data + λ * loss_phys\n",
        "end\n"
    ]
}

cell_train = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "loss_history = Float32[]\n",
        "callback_fn = function (θ, loss_val)\n",
        "    push!(loss_history, loss_val)\n",
        "    if length(loss_history) % 10 == 0\n",
        "        println(\"Iteration: $(length(loss_history)) | Loss: $(loss_val)\")\n",
        "    end\n",
        "    return false\n",
        "end\n",
        "\n",
        "optf = OptimizationFunction(total_loss, Optimization.AutoZygote())\n",
        "optprob = OptimizationProblem(optf, p_initial)\n",
        "\n",
        "println(\"Starting Training...\")\n",
        "# Train for 10 epochs to verify loss convergence quickly\n",
        "result = solve(optprob, Adam(0.01), maxiters = 10, callback = callback_fn)\n",
        "println(\"Training Complete!\")\n"
    ]
}

nb['cells'].extend([
    {"cell_type": "markdown", "metadata": {}, "source": ["# PI-NODE-SR Refactoring (Optimized for GPU/Float32)"]},
    cell_deps, cell_params, cell_rates, cell_hh, 
    cell_nn, cell_neural_dyn, cell_training_data, cell_scale_factors,
    cell_predict, cell_physics_loss, cell_total_loss, cell_train
])

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")

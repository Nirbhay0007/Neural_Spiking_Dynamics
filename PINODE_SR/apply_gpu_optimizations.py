import json
import os

notebook_path = r'C:\Users\ADMIN\Downloads\Neural_Spiking_Dynamics\PINODE_SR\pinode_sr.ipynb'

if not os.path.exists(notebook_path):
    print(f"Notebook not found at: {notebook_path}")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] != 'code':
        continue
        
    source = "".join(cell['source'])
    
    # 1. Update Training Data to be on GPU
    if "t_train =  Float32.(df_ordered.timestamp)" in source and "z_train =" in source:
        cell['source'] = [
            "t_train_cpu = Float32.(df_ordered.timestamp)\n",
            "z_train_cpu = Float32.(Matrix(df_ordered[:, [:V, :n, :m, :h]])')\n",
            "\n",
            "# Move training data to GPU\n",
            "t_train = t_train_cpu |> gpu_dev\n",
            "z_train = z_train_cpu |> gpu_dev\n"
        ]

    # 2. Update Scale Factors calculation
    elif "scale_factors = calculate_scale_factors(z_train, t_train)" in source:
        cell['source'] = [
            "# Run this before your optimization loop (Compute on CPU, move to GPU)\n",
            "scale_factors = calculate_scale_factors(z_train_cpu, t_train_cpu) |> gpu_dev\n"
        ]

    # 3. Update Lux Network & Parameters
    elif "ps,st = Lux.setup(rng,nn)" in source and "p_initial = ComponentVector(ps) |> Lux.f32" in source:
        cell['source'] = [
            "nn = Lux.Chain(\n",
            "    Lux.Dense(5 => 64, Lux.tanh),\n",
            "    Lux.LayerNorm(64),\n",
            "    Lux.Dense(64 => 64, Lux.tanh),\n",
            "    Lux.LayerNorm(64),\n",
            "    Lux.Dense(64 => 4)\n",
            ")\n",
            "rng = Random.default_rng()\n",
            "ps, st = Lux.setup(rng, nn)\n",
            "\n",
            "# --- Move Network to GPU ---\n",
            "ps_gpu = ps |> gpu_dev\n",
            "st_gpu = st |> gpu_dev\n",
            "\n",
            "# Flatten parameters for the optimizer\n",
            "p_flat, reconstruct = Optimisers.destructure(ps_gpu)\n",
            "p_initial = Float32.(p_flat)\n"
        ]

    # 4. Update Physics Loss
    elif "function pyhysics_loss(nn, state_data, time_data,θ, scale_factors,reconstruct)" in source:
        cell['source'] = [
            "function pyhysics_loss(nn, state_data, time_data, θ, scale_factors, reconstruct)\n",
            "    ps_current = reconstruct(θ)\n",
            "    \n",
            "    # Reshape time_data to 1xN matrix to match GPU dimensions\n",
            "    nn_inputs = vcat(state_data, reshape(time_data, 1, :))\n",
            "\n",
            "    # Evaluate neural network on GPU\n",
            "    pred_derivs, _ = nn(nn_inputs, ps_current, st_gpu)\n",
            "\n",
            "    # calculate the HH derivatives\n",
            "    true_dervis = hh_equations(state_data)\n",
            "\n",
            "    # compute scale-aware residual\n",
            "    residuals = (pred_derivs .- true_dervis) ./ scale_factors\n",
            "    \n",
            "    return mean(residuals.^2)\n",
            "end\n"
        ]

    # 5. Update neural_dynamics and ODEProblem
    elif "function neural_dynamics(u, p, t)" in source and "node_prob = ODEProblem" in source:
        cell['source'] = [
            "# Float32 Rate Functions\n",
            "α_m(V) = 0.1f0 .* (V .+ 40.0f0) ./ (1.0f0 .- exp.(-(V .+ 40.0f0) ./ 10.0f0))\n",
            "β_m(V) = 4.0f0 .* exp.(-(V .+ 65.0f0) ./ 18.0f0)\n",
            "α_h(V) = 0.07f0 .* exp.(-(V .+ 65.0f0) ./ 20.0f0)\n",
            "β_h(V) = 1.0f0 ./ (1.0f0 .+ exp.(-(V .+ 35.0f0) ./ 10.0f0))\n",
            "α_n(V) = 0.01f0 .* (V .+ 55.0f0) ./ (1.0f0 .- exp.(-(V .+ 55.0f0) ./ 10.0f0))\n",
            "β_n(V) = 0.125f0 .* exp.(-(V .+ 65.0f0) ./ 80.0f0)\n",
            "\n",
            "function neural_dynamics(u, p, t)\n",
            "    ps_structured = reconstruct(p)\n",
            "    \n",
            "    # Wrap scalar 't' in a GPU array before concatenating\n",
            "    t_arr = fill(t, 1) |> gpu_dev\n",
            "    raw_input = vcat(u, t_arr) \n",
            "    \n",
            "    input_2d = reshape(raw_input, :, 1)\n",
            "    \n",
            "    # Evaluate on GPU\n",
            "    dudt_2d, _ = nn(input_2d, ps_structured, st_gpu) \n",
            "    \n",
            "    return vec(dudt_2d)\n",
            "end\n",
            "\n",
            "p = Float32[\n",
            "    10.0,  # External current\n",
            "    120.0, # Max Sodium conductance\n",
            "    36.0,  # Max Potassium conductance\n",
            "    0.3,   # Leak conductance\n",
            "    50.0,  # Sodium reversal potential\n",
            "    -77.0, # Potassium reversal potential\n",
            "    -54.4, # Leak reversal potential\n",
            "    1.0    # Membrane capacitance\n",
            "]\n",
            "\n",
            "# Move Initial Conditions to GPU\n",
            "u0_gpu = Float32[-65.0f0, 0.05f0, 0.6f0, 0.32f0] |> gpu_dev\n",
            "tspan = (0.0f0, 50.0f0)\n",
            "node_prob = ODEProblem(neural_dynamics, u0_gpu, tspan, p_initial)\n"
        ]

    # 6. Fix neonorange color in Plotting
    elif "color = :neonorange" in source:
        new_source = []
        for line in cell['source']:
            if "color = :neonorange" in line:
                new_source.append(line.replace("color = :neonorange", "color = :orange"))
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully updated pinode_sr.ipynb with GPU optimizations!")

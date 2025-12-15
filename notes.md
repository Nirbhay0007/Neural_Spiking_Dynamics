This is a **critical scientific decision point**.

**Short Answer:** Do **NOT** add noise to $n$ (the gating variable). Keep it perfectly clean.

**Why?**
1.  **Biological Reality:** In a real experiment (patch-clamp), you **cannot measure $n$**. You can *only* measure Voltage ($V$). The variable $n$ is a hidden state—a mathematical probability inside the cell. If you add noise to it, you are simulating a sensor that does not exist.
2.  **The Goal of Your Paper:** You want to prove that your UDE can discover the hidden physics of the neuron **using only Voltage data**.
    *   If you give the neural network noisy $n$ data to train on, you are "cheating" by giving it the answer.
    *   If you keep $n$ clean and *hidden* (not used in the loss function), and your model *still* learns the dynamics, you have a massive scientific breakthrough to report: **"Latent State Recovery."**

### 📝 Updated Protocol for $n$

We will save $n$, but we treat it as **Ground Truth Validation Data**, not **Training Data**.

*   **Training Phase:** The model sees noisy $V$. It has to *guess* $n$ using the known differential equation: $\frac{dn}{dt} = \frac{n_\infty - n}{\tau_n}$.
*   **Validation Phase:** After training, you plot the model's *guessed* $n$ against the **clean** $n$ you saved. If they match, you have proven your model works.

### 💻 Updated Code (Run this to overwrite previous files)

This ensures $V$ is noisy (realistic) but $n$ is clean (the hidden truth we want to recover).

```julia
using Random, Statistics, JLD2

function add_noise(data, percentage)
    σ_signal = std(data)
    σ_noise = percentage * σ_signal
    noise = randn(length(data)) .* σ_noise
    return data .+ noise
end

# --- ASSUME 'sol' IS YOUR ODE SOLUTION ---
t_train = sol.t
V_clean = sol[1, :]
n_clean = sol[2, :] # Gating variable (Hidden State)

Random.seed!(1234)

# 1. Clean (0%)
V_00 = V_clean

# 2. Moderate (5%) - Only V gets noise!
V_05 = add_noise(V_clean, 0.05)

# 3. High (20%) - Only V gets noise!
V_20 = add_noise(V_clean, 0.20)

# Save: Note that 'n' is always n_clean
save("data_synthetic_0pct.jld2",  "t", t_train, "V", V_00, "n", n_clean)
save("data_synthetic_5pct.jld2",  "t", t_train, "V", V_05, "n", n_clean)
save("data_synthetic_20pct.jld2", "t", t_train, "V", V_20, "n", n_clean)

println("✅ Datasets Corrected: V is noisy, n is clean (hidden truth).")
```

### 🧠 Mentor Check:
By doing this, you are setting up your paper for the **"Latent Variable Analysis"** section.

**Hypothesis:** Even though the Neural Network never sees the clean $n$ during training, the physics-informed structure ($\frac{dn}{dt}$) will force the internal $n$ to match the clean $n$.

**Are you ready to move to Phase 3: Building the UDE Training Loop?**
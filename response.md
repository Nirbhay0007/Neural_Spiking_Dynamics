### A. Research Question & Hypothesis

**Problem Statement:** While the Hodgkin-Huxley model elegantly describes neural spikes via voltage-dependent gating kinetics driven by diffusion and electrostatic forces ($\alpha$ and $\beta$ rates), identifying unobserved, highly nonlinear macroscopic ionic currents (e.g., $I_{Na}$) from scalar membrane voltage recordings remains an ill-posed inverse problem. Traditional parameter fitting struggles with the stiff nature of these differential equations, while pure Machine Learning surrogates ignore decades of established biophysical equilibrium thermodynamics ($n_\infty, \tau_n$).

**Hypothesis:** Universal Differential Equations (UDEs) can accurately discover latent ionic currents by embedding Multi-Layer Perceptrons directly into structurally reduced, biophysically rigorous neuronal ODEs (e.g., a 2D HH reduction leveraging known $K^+$ and Leak properties). This hybrid architecture out-generalizes purely data-driven methods by restricting the hypothesis space to physically valid trajectories using explicit stiff solvers and multistage optimization.


### B. Methodology & Architecture

**Model Type:** Universal Differential Equation (UDE) leveraging Julia's SciML ecosystem constraint by rigorous first-principles electrophysiology.

**Key Architectural Components:**
- **Physics Backbone:** A mathematically reduced 2D Hodgkin-Huxley system. Rather than full 4D tracking, the model elegantly assumes fast sodium activation and tied inactivation ($m \approx m_\infty(V), h \approx h_\infty(V)$), reducing the continuous-time dynamics to just Voltage ($V$) and Potassium activation ($n$). The capacitance equation $C_m \frac{dV}{dt} = \sum I_{ion}$ explicitly grounds the network.
- **Neural Component:** A dense MLP (`[V/100, n] -> 32 -> 32 -> 1`) using continuous `tanh` activations to predict the highly non-linear $I_{Na}$ residual. Note the explicit scaling ($V/100$ input, $pred \times 100$ output) to maintain stable gradients during backpropagation.
- **Optimization Strategy:** Unlike standard Deep Learning, our setup requires stiff ODE integration. We utilize `Rodas5` for stable forward integration coupled with `InterpolatingAdjoint(autojacvec=ZygoteVJP())` for memory-efficient gradient calculation. 

**Multiphasic Training Pipeline:**
Training stiff UDEs fundamentally breaks standard Adam. Therefore, a robust 3-Phase optimization strategy is implemented:
1. **Exploration (AdamW, lr=0.005):** Fast navigation of the high-loss landscape to reach a stable "bouncing state".
2. **Stabilization (AdamW, lr=0.01):** Settle the oscillating gradients and capture the macroscopic spike frequency.
3. **Precision Polish (L-BFGS):** Utilizing `OptimizationOptimJL.BFGS()` to apply Newton-based second-order optimization, perfectly capturing the strict amplitude and phase of the threshold dynamics.

**Code Snippets:**

*Neural Network & Scaling Logic:*
```julia
U = Lux.Chain(
    Lux.Dense(2 => 32, tanh),
    Lux.Dense(32 => 32, tanh),
    Lux.Dense(32 => 1)
) |> f64
```

*Physics Enforcement in ODE:*
```julia
function hodgkin_huxley_UDE!(du, u, p, t)
    V, n = u  
    
    # Scaled Neural Network predicts unobserved Sodium Current
    Input_V = V/100
    pred = U([V, n], p, st)[1][1]  
    pred_I_Na = pred * 100
    
    # Known physics: Potassium & Leak currents tied to Nernst potentials
    I_K = g_K * (n^4) * (V - E_K)
    I_L = g_L * (V - E_L)
    I_ext = Stimulus(t) 

    # Dynamics explicitly enforce C_m(dV/dt) = I
    du[1] = (I_ext - pred_I_Na - I_K - I_L) / Cm
    
    # Potassium gate exponential relaxation dynamics
    du[2] = (n_inf(V) - n) / tau_n(V)
end
```


### C. Baselines & Metrics

**Baselines:**
1. **Standard PINN (Physics-Informed Neural Network):** Evaluated to demonstrate that soft-penalizing the ODE residual in the loss function fails catastrophically against stiff gating variable kinetics ($\tau_n, n_\infty$).
2. **Classical Parameter Fitting:** Traditional least-squares approximation on standard $g_{Na}, E_{Na}$ targets.
3. **Data-Driven Neural ODE:** A fully unconstrained neural surrogate mapping time -> voltage without hardcoded Potassium or Leak biophysics.

**Metrics:**
- **L2 Norm (Mean Squared Error):** Calculated strictly on the macroscopic membrane voltage (`pred_V .- V`) focusing only on observable variables mimicking real patch-clamp constraints.
- **Phase/Timing Error:** Captured using Victor-Purpura or Van Rossum metrics, crucial for testing if the UDE maintains the correct excitability threshold and refractory periods over extended firing regimes.
- **Latent Variable Recovery Error:** Ground truth comparison of the neural output `pred_I_Na` against the true mathematical sodium current during our synthetic benchmarking phase.


### D. Data Generation

**Data Source:** A structured transition from pristine biological computation to empirical observation.

Our curriculum generation ensures strict validation of the continuous-time backward passes before tackling real biological noise.
- **Phase 1 (Synthetic Formulation & Validation):** Ground truth data (`noise_0_hh_2d_model.jld`) is generated computationally via `DifferentialEquations.jl` utilizing the ultra-precise `Rodas5P` solver (tolerances at 1e-10). The simulation defines the 2D abstraction with explicit Hodgkin-Huxley alpha/beta formulation, producing a pristine $[V(t), n(t)]$ trajectory. This verifies the L-BFGS/Adjoint recovery mechanics.
- **Phase 2 (In-Vitro Ingestion):** Moving past theoretical constructs, we deploy `pynwb` to extract experimental patch-clamp datasets directly from the Allen Institute Cell Types Database (e.g., recording 324257144 under specific "Noise 1" sweep protocols). This subjects the UDE to genuine recording artifacts, stochastic channel noise, and unmodeled ion variations (e.g., Calcium channels), testing the network's capacity to absorb residual unknown electrophysiology into the surrogate $I_{Na}$ representation.

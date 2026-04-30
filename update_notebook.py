import nbformat

notebook_path = r'c:\Users\nirbh\Neural_Spiking_Dynamics\Hudgin_Huxely_4D_model.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

julia_code = """using DataFrames, CSV, Random, Statistics

# Set seed for reproducibility
Random.seed!(42)

# Generate noise function
function apply_noise(clean_data, noise_pct)
    if noise_pct == 0
        return clean_data
    end
    sigma = std(clean_data) * (noise_pct / 100.0)
    return clean_data .+ randn(length(clean_data)) .* sigma
end

# Extract 4D data from the solution
t_data = sol.t
V_clean = sol[1, :]
m_clean = sol[2, :]
h_clean = sol[3, :]
n_clean = sol[4, :]

out_base = "c:/Users/nirbh/Neural_Spiking_Dynamics/notebooks/1_data_generation"

# Apply noise and save
for p in [0, 10, 20]
    df = DataFrame(
        t = t_data,
        V = apply_noise(V_clean, p),
        m = apply_noise(m_clean, p),
        h = apply_noise(h_clean, p),
        n = apply_noise(n_clean, p)
    )
    folder_path = joinpath(out_base, "$(p)_noise")
    mkpath(folder_path) # Create directory if it doesn't exist
    CSV.write(joinpath(folder_path, "HH_4D_data.csv"), df)
    println("Saved $(p)% noise data to: $folder_path")
end
println("All data generated and saved successfully!")
"""

# Check if the code is already there to avoid duplicates
cell_exists = any(cell.source.strip() == julia_code.strip() for cell in nb.cells)

if not cell_exists:
    # Append the markdown cell
    markdown_cell = nbformat.v4.new_markdown_cell(source="### 💾 Noise Generation and Data Export\nThis cell extracts the 4D state variables ($V, m, h, n$), calculates their variance, applies 0%, 10%, and 20% Additive White Gaussian Noise (AWGN), and saves them to separate directories.")
    nb.cells.append(markdown_cell)
    
    # Append the code cell
    new_cell = nbformat.v4.new_code_cell(source=julia_code)
    nb.cells.append(new_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Notebook updated successfully.")
else:
    print("Cell already exists in the notebook.")

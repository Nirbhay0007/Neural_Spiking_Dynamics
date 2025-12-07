using JLD2

# Specify the path to your .jld or .jld2 file
file_path = "neuron_mission_log.jld" # Confirmed correct path and filename

# Open the file and read its contents
try
    data = jldopen(file_path, "r") do file
        available_keys = keys(file)
        println("Keys in the JLD2 file: ", available_keys)

        # Access the 'params' and 'loss_history'
        loaded_params = file["params"]
        loaded_loss_history = file["loss_history"]
        
        println("\nSuccessfully loaded 'params' (first 5 elements): ", loaded_params[1:min(5, end)])
        println("Successfully loaded 'loss_history' (last 5 elements): ", loaded_loss_history[max(1, end-4):end])

        # You can return these variables or do further processing
        return (params=loaded_params, loss_history=loaded_loss_history)
    end
    println("\nSuccessfully processed file. Loaded data summary:")
    println("  Parameters (length): ", length(data.params))
    println("  Loss History (length): ", length(data.loss_history))
    println("  Last Loss: ", last(data.loss_history))

catch e
    println("Error loading JLD2 file: ", e)
end
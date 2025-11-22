using JLD2

# Specify the path to your .jld or .jld2 file
file_path = "neuron_mission_log.jld" # or "your_file_name.jld"

# Open the file and read its contents
try
    data = jldopen(file_path, "r") do file
        # You can access variables stored in the file like a dictionary
        # For example, if you saved a variable named 'my_variable'
        my_variable = file["checkpoint_data"]
        
        # If you want to see all keys (variable names) in the file:
        println("Keys in the JLD2 file: ", keys(file))

        # You can return specific data or the file handle itself if needed
        return my_variable # Or a Dict of all variables you want
    end
    println("Successfully loaded data: ", data)
catch e
    println("Error loading JLD2 file: ", e)
end
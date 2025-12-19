using JLD2
data = load("ude_best_model.jld2")
data["best_loss"]
length(data["loss_history"])




# -----------------

data1= load("Data/synthetic_data/noise_0_hh_2d_model.jld")
function lengfind(data)
          println((data))
          
end

lengfind(data1["t"])
lengfind(data1["V"][1:20])
lengfind(data1["n"])


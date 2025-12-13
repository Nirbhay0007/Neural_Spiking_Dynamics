using DataFrames
using CSV
using Plots

df = CSV.read("C:/Users/Admin/Downloads/Neural_Spiking_Dynamics/expermental_data/neuron_data_noise1.csv", DataFrame)



# WE are going to find the first spike V above -20mV
spike_index = findfirst(df.voltage_mv.> -20)